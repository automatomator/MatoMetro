# src/analysis_worker.py

import numpy as np
import trimesh
import cv2
from scipy.spatial import ConvexHull, KDTree
from scipy.ndimage import rotate as ndimage_rotate
from PyQt6.QtCore import QThread, pyqtSignal
from .utils import logger # Ensure this import is correct for your logger

# --- Helper function for consistent angle from minAreaRect ---
def get_min_area_rect_angle(contour):
    """
    Calculates the orientation angle of a contour using its minimum area rectangle.
    The angle returned is the rotation needed to make the longer side of the rectangle horizontal.
    """
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    if rect[1][0] < rect[1][1]: # width < height
        angle = 90 + angle
    return angle

class AnalysisWorker(QThread):
    """
    A QThread to perform the analysis in the background,
    preventing the UI from freezing. It emits signals to update the UI
    with progress, results, or errors.
    """
    analysis_completed = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)
    progress_updated_with_text = pyqtSignal(int, str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_path = config.get('image_path')
        self.reference_path = config.get('reference_path')
        self.reference_type = config.get('reference_type')
        self.projection_axis = config.get('projection_axis') # Kept for future STL support

        # Initialize image processing parameters with defaults and log them
        self.threshold_value = config.get('threshold_value', 127)
        logger.info(f"Initialized threshold_value: {self.threshold_value}")

        self.canny_threshold1 = config.get('canny_threshold1', 50)
        logger.info(f"Initialized canny_threshold1: {self.canny_threshold1}")

        self.canny_threshold2 = config.get('canny_threshold2', 150)
        logger.info(f"Initialized canny_threshold2: {self.canny_threshold2}")

        self.blur_kernel = config.get('blur_kernel', 5)
        logger.info(f"Initialized blur_kernel: {self.blur_kernel}")

        self.processing_method = config.get('processing_method', 'Canny')
        logger.info(f"Initialized processing_method: {self.processing_method}")


    def _process_image_for_contour(self, image_path, is_part_image=True):
        logger.info(f"Loading image: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")

        logger.info(f"Applying blur with kernel: {self.blur_kernel}")
        # Apply Gaussian blur
        if self.blur_kernel > 0: # Only apply if kernel is positive
            # Ensure kernel size is odd for GaussianBlur
            kernel_size = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
            if kernel_size == 0: kernel_size = 1 # Avoid 0 kernel size even after adjustment if blur_kernel was 0
            image_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            logger.info(f"Applied Gaussian blur with kernel size: {kernel_size}")
        else:
            image_blurred = image
            logger.info("No blur applied (kernel size is 0).")

        logger.info(f"Processing method: {self.processing_method}")
        if self.processing_method == 'Canny':
            # Apply Canny edge detection
            edges = cv2.Canny(image_blurred, self.canny_threshold1, self.canny_threshold2)
            processed_image = edges
        else: # Default to Threshold
            # Apply binary thresholding
            # Using THRESH_OTSU for automatic thresholding if desired, otherwise THRESH_BINARY
            _, binary_image = cv2.threshold(image_blurred, self.threshold_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_image = binary_image

        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError(f"No contours found in {image_path}. Try adjusting image processing parameters.")

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        centroid = self._get_contour_centroid(largest_contour)
        logger.info(f"{'Part' if is_part_image else 'Reference'} contour extracted with centroid at {centroid}.")
        return largest_contour, centroid, image.shape # Return original image shape for canvas sizing

    def _get_contour_centroid(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0: # Avoid division by zero
            return (0, 0)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    def _rotate_contour(self, contour, angle, center):
        """Rotates a contour around a given center point."""
        # Ensure contour is float32 for getRotationMatrix2D and transform
        contour_np = contour.squeeze().astype(np.float32)

        M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        rotated_contour = cv2.transform(np.array([contour_np]), M)[0]

        return rotated_contour.reshape((-1, 1, 2)).astype(np.int32)

    def _translate_contour(self, contour, offset):
        """Translates a contour by a given offset."""
        return contour + np.array(offset).astype(np.int32) # Ensure int32 for contour points

    def _align_contours_icp(self, source_contour, target_contour, iterations=100, tolerance=1.0):
        """
        Aligns a source contour to a target contour using an ICP-like approach
        with `cv2.estimateAffine2D`. This method finds translation, rotation, and scaling.

        Args:
            source_contour (np.array): The contour to be transformed (e.g., part contour).
            target_contour (np.array): The reference contour (e.g., CAD screenshot contour).
            iterations (int): Maximum number of ICP iterations.
            tolerance (float): Convergence threshold for transformation change (norm of matrix difference).

        Returns:
            np.array: The aligned source contour.
            np.array: The transformation matrix (2x3 affine matrix).
        """
        logger.info(f"Starting ICP-like alignment with {iterations} iterations and tolerance {tolerance}.")

        aligned_source_contour = source_contour.astype(np.float32).squeeze()
        target_points = target_contour.astype(np.float32).squeeze()

        if len(aligned_source_contour) < 3 or len(target_points) < 3:
            logger.warning("Not enough points for affine transformation estimation. Skipping ICP.")
            # Return source contour as is and identity matrix
            return source_contour, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # Build KDTree for efficient nearest neighbor search on the target contour
        kdtree = KDTree(target_points)

        current_transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) # Identity matrix for initial transform

        for i in range(iterations):
            self.progress_updated_with_text.emit(int(70 + 20 * (i / iterations)), f"Aligning (ICP Iteration {i+1}/{iterations})...")
            
            # Step 1: Find Correspondences (nearest neighbors)
            distances, indices = kdtree.query(aligned_source_contour)
            corresponding_target_points = target_points[indices]

            # Step 2: Estimate Transformation (using estimateAffine2D)
            # Use adjusted RANSAC parameters if needed for robustness
            M, inliers = cv2.estimateAffine2D(aligned_source_contour, corresponding_target_points,
                                            # method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=2000
                                            )

            if M is None:
                logger.warning("cv2.estimateAffine2D returned None. Check contour quality or adjust parameters.")
                break # Exit loop if transformation estimation fails

            # Check for convergence (change in transformation)
            # Compare only the transformation part that would cause point movement (translation and rotation components)
            # For scale, comparing determinant or frobenius norm of rotation/scale part might be better
            # For simplicity, comparing the entire matrix norm is usually sufficient for convergence
            transform_diff = np.linalg.norm(M - current_transform)
            current_transform = M # Update current transform for next iteration

            if transform_diff < tolerance:
                logger.info(f"ICP converged at iteration {i+1}.")
                break
            
            # Step 3: Apply Transformation
            aligned_source_contour = cv2.transform(np.array([aligned_source_contour]), M)[0]

        logger.info("ICP-like alignment completed.")
        return aligned_source_contour.reshape((-1, 1, 2)).astype(np.int32), current_transform


    def run(self):
        """
        Executes the image and reference (Image) analysis,
        performing automatic scale, translation, and rotation alignment.
        """
        try:
            self.progress_updated_with_text.emit(5, "Starting Analysis...")
            
            # --- Process Part Image ---
            self.progress_updated_with_text.emit(20, "Processing part image...")
            part_contour_orig, part_centroid_orig, part_image_shape = self._process_image_for_contour(self.image_path, is_part_image=True)
            
            # --- Process Reference Image ---
            self.progress_updated_with_text.emit(30, "Processing reference image...")
            # For now, only 'image' type is supported for direct contour alignment
            if self.reference_type != 'image':
                raise ValueError("Only 'image' reference type is currently supported for alignment.")
            
            ref_contour_orig, ref_centroid_orig, ref_image_shape = self._process_image_for_contour(self.reference_path, is_part_image=False)
            
            # --- Create Original Reference Projection Image (for display) ---
            # This image will show the reference contour as extracted
            reference_projection_image_orig_display = np.full(ref_image_shape, 255, dtype=np.uint8) # White background
            cv2.drawContours(reference_projection_image_orig_display, [ref_contour_orig], -1, 0, 3) # Black contour, thicker
            logger.info("Image reference processed.")
            self.progress_updated_with_text.emit(40, "Reference processed.")
            
            # --- Initial Centroid Matching (Translation) ---
            logger.info("Performing initial centroid matching (translation).")
            self.progress_updated_with_text.emit(50, "Aligning centroids...")

            # Translate both contours to origin for consistent alignment
            part_contour_translated_to_origin = self._translate_contour(part_contour_orig, -np.array(part_centroid_orig))
            ref_contour_translated_to_origin = self._translate_contour(ref_contour_orig, -np.array(ref_centroid_orig))

            # --- NO EXPLICIT AREA-BASED SCALING HERE ---
            # ICP will handle the scaling directly.

            # --- Perform ICP Alignment ---
            self.progress_updated_with_text.emit(70, "Performing iterative contour alignment (ICP)...")
            
            # Pass the origin-translated contours directly. ICP will find optimal T, R, S.
            aligned_part_contour_relative, icp_transform_matrix = self._align_contours_icp(
                part_contour_translated_to_origin, ref_contour_translated_to_origin
            )
            self.progress_updated_with_text.emit(90, "ICP alignment complete. Generating comparison images.")

            # --- Prepare Contours for Display ---
            # To display them superimposed on a single canvas, we need to translate them
            # from their relative-to-origin positions to a common display center.
            
            # Use the part image's original dimensions for the canvas size
            display_height, display_width = part_image_shape[:2]
            display_center_x, display_center_y = display_width // 2, display_height // 2

            # Calculate offset to move the (0,0) origin of the aligned contours to the display center
            offset_to_display_center = np.array([display_center_x, display_center_y])

            # Apply this offset to both the ICP-aligned part contour and the reference contour
            # The reference contour was only translated to origin, now translate it back to display center
            final_aligned_part_contour_display = self._translate_contour(aligned_part_contour_relative, offset_to_display_center)
            final_aligned_ref_contour_display = self._translate_contour(ref_contour_translated_to_origin, offset_to_display_center)

            # --- Create Superimposed Image for Visualization ---
            superimposed_image = np.full((display_height, display_width, 3), 255, dtype=np.uint8) # White background
            
            # Draw the final aligned contours
            cv2.drawContours(superimposed_image, [final_aligned_part_contour_display], -1, (0, 255, 0), 3) # Green for part, thicker
            cv2.drawContours(superimposed_image, [final_aligned_ref_contour_display], -1, (0, 0, 255), 3) # Red for reference, thicker

            # --- Calculate Area and Deviation (Pixel-based, then convert to pseudo-mm) ---
            
            x,y,w,h = cv2.boundingRect(ref_contour_orig)
            ref_diagonal_pixels = np.sqrt(w**2 + h**2)
            
            # Pseudo-scale for "mm" conversion. This assumes ref_diagonal_pixels corresponds to 100 "demo_mm".
            if ref_diagonal_pixels > 0:
                pseudo_pixels_per_mm = ref_diagonal_pixels / 100.0
                logger.info(f"Pseudo-scale for mm conversion: {pseudo_pixels_per_mm:.2f} pixels/mm (based on ref diagonal = 100mm)")
            else:
                pseudo_pixels_per_mm = 1.0
                logger.warning("Reference diagonal is zero, pseudo_pixels_per_mm set to 1.0.")

            # Calculate areas of the *final aligned* contours
            image_area_after_alignment_pixels = cv2.contourArea(final_aligned_part_contour_display)
            reference_area_after_alignment_pixels = cv2.contourArea(final_aligned_ref_contour_display)

            image_area_mm2 = image_area_after_alignment_pixels / (pseudo_pixels_per_mm ** 2)
            reference_area_mm2 = reference_area_after_alignment_pixels / (pseudo_pixels_per_mm ** 2)

            area_deviation_percent = abs(image_area_mm2 - reference_area_mm2) / reference_area_mm2 * 100 if reference_area_mm2 != 0 else 0

            # --- Max Deviation in MM ---
            max_deviation_mm = 0.0
            
            # Ensure enough points for KDTree
            part_points_float = final_aligned_part_contour_display.squeeze().astype(np.float32)
            ref_points_float = final_aligned_ref_contour_display.squeeze().astype(np.float32)

            if len(part_points_float) > 0 and len(ref_points_float) > 0:
                kdtree_ref = KDTree(ref_points_float)
                kdtree_part = KDTree(part_points_float)

                distances_part_to_ref, _ = kdtree_ref.query(part_points_float)
                distances_ref_to_part, _ = kdtree_part.query(ref_points_float)
                
                max_deviation_pixels = max(np.max(distances_part_to_ref), np.max(distances_ref_to_part))
                max_deviation_mm = max_deviation_pixels / pseudo_pixels_per_mm
            else:
                logger.warning("Contours are empty after alignment, cannot calculate max deviation.")
                max_deviation_mm = 0.0


            # --- Prepare `processed_image` (Part Boundary Display) ---
            # Draw the original part contour on a white background with thicker lines
            boundary_image = np.full(part_image_shape, 255, dtype=np.uint8) # White background
            cv2.drawContours(boundary_image, [part_contour_orig], -1, 0, 3) # Black contour, thickness 3

            results = {
                'image_area_mm2': image_area_mm2,
                'reference_area_mm2': reference_area_mm2,
                'area_deviation_percent': area_deviation_percent,
                'max_deviation_mm': max_deviation_mm,
                'processed_image': boundary_image, # Now has thicker black contour
                'projected_stl_image': reference_projection_image_orig_display, # Already made thicker
                'superimposed_image': superimposed_image # Already made thicker
            }
            self.progress_updated_with_text.emit(100, "Analysis Complete!")
            logger.info("Analysis completed successfully.")
            self.analysis_completed.emit(results)

        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}", exc_info=True)
            self.analysis_error.emit(f"An error occurred: {e}")