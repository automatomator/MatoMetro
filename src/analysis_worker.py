# src/analysis_worker.py
import numpy as np
import trimesh
import cv2
from scipy.spatial import ConvexHull, KDTree
from scipy.ndimage import rotate as ndimage_rotate
from PyQt6.QtCore import QThread, pyqtSignal
import tempfile
import os

from .utils import logger
from .image_processing import ImageProcessor # Import the ImageProcessor
from .config_models import AnalysisConfig # Import the dataclass

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
    return angle % 180 # Normalize angle to be between 0 and 180

class AnalysisWorker(QThread):
    """
    A QThread to perform the analysis in the background,
    preventing the UI from freezing. It emits signals to update the UI
    with progress, results, or errors.
    """
    analysis_completed = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)
    progress_updated_with_text = pyqtSignal(int, str)

    def __init__(self, config: AnalysisConfig):
        super().__init__()
        self.config = config
        self.image_processor = ImageProcessor(config) # Initialize ImageProcessor here

    def run(self):
        logger.info("Analysis started.")
        self.progress_updated_with_text.emit(0, "Starting analysis...")

        try:
            image_processor = self.image_processor

            # --- Step 1: Process Part Image ---
            self.progress_updated_with_text.emit(10, "Processing part image...")
            part_original_image, part_contour, part_pixels_per_mm = image_processor.process_image(
                self.config.image_path, self.config.part_pixels_per_mm_value
            )
            if part_contour is None or len(part_contour) < 3:
                raise ValueError("Could not detect sufficient contour points in the part image. Please check image or adjust parameters.")
            logger.info(f"Part image processed. Pixels/mm: {part_pixels_per_mm}")


            # --- Step 2: Process Reference (Image or STL) ---
            self.progress_updated_with_text.emit(30, "Processing reference...")
            reference_original_image, reference_contour, reference_pixels_per_mm = image_processor.process_reference(
                self.config.reference_path, self.config.reference_type, self.config.reference_pixels_per_mm_value,
                self.config.projection_axis
            )
            if reference_contour is None or len(reference_contour) < 3:
                raise ValueError("Could not detect sufficient contour points in the reference. Please check file or adjust parameters.")
            logger.info(f"Reference processed. Type: {self.config.reference_type}, Pixels/mm: {reference_pixels_per_mm}")

            # --- Step 3: Alignment and Deviation Calculation ---
            self.progress_updated_with_text.emit(50, "Performing alignment...")

            # Ensure contours are Nx2 for calculations and then convert to float32
            # The contours from cv2.findContours are typically Nx1x2.
            # Reshape them to Nx2 for easier arithmetic operations.
            part_contour_flat = part_contour.reshape(-1, 2).astype(np.float32)
            reference_contour_flat = reference_contour.reshape(-1, 2).astype(np.float32)

            # Explicit check for sufficient points after flattening
            if part_contour_flat.shape[0] < 3:
                raise ValueError("Part contour has fewer than 3 points after flattening, cannot perform alignment.")
            if reference_contour_flat.shape[0] < 3:
                raise ValueError("Reference contour has fewer than 3 points after flattening, cannot perform alignment.")


            # Calculate scaling factor to bring part contour to reference's pixel density
            # This factor is used to scale the part contour so that its "pixels"
            # represent the same real-world distance as the reference's "pixels".
            # This is crucial for estimateAffine2D to find a meaningful transformation.
            # Example: if part is 0.22 px/mm and ref is 100 px/mm, ref is much denser.
            # We scale part_contour by (100 / 0.22) = ~454.54 to make it comparable.
            alignment_scale_factor = reference_pixels_per_mm / part_pixels_per_mm
            
            # Apply scaling to the part contour (convert part's pixel space to reference's pixel space)
            part_aligned_scale_contour = part_contour_flat * alignment_scale_factor

            # Center both contours around their respective centroids
            # This helps in providing stable input for estimateAffine2D
            part_centroid = np.mean(part_aligned_scale_contour, axis=0)
            reference_centroid = np.mean(reference_contour_flat, axis=0)

            part_scaled_centered = part_aligned_scale_contour - part_centroid
            reference_centered_final = reference_contour_flat - reference_centroid # Renamed for clarity

            # Final reshape for cv2.estimateAffine2D (expects Nx1x2)
            # This format is standard for many OpenCV geometry functions.
            part_scaled_for_affine = part_scaled_centered.reshape(-1, 1, 2)
            reference_centered_for_affine = reference_centered_final.reshape(-1, 1, 2)

            # Perform alignment using estimateAffine2D (finds rotation, translation, scale, shear)
            # method=cv2.RANSAC makes it robust to outliers.
            # ransacReprojThreshold: Maximum allowed reprojection error to treat a point pair as an inlier.
            logger.info("Attempting to estimate affine transformation...")
            M, inliers = cv2.estimateAffine2D(
                part_scaled_for_affine,
                reference_centered_for_affine,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )

            # Check if affine transformation matrix was successfully estimated
            if M is None:
                raise RuntimeError(
                    "Failed to estimate affine transformation. "
                    "This might indicate insufficient matching features or highly dissimilar contours. "
                    "Consider adjusting image processing parameters or checking image quality, or try a different reference."
                )
            
            logger.info("Affine transformation estimated successfully.")

            # Apply the estimated transformation to the part contour (still in reference's pixel scale)
            aligned_part_contour_transformed = cv2.transform(part_scaled_for_affine, M)

            # Convert the aligned part contour back to the 'real-world' mm scale if needed for deviation.
            # Since reference_contour_flat is already in 'reference pixels', and we applied M to a
            # scaled part contour (now effectively in 'reference pixels'), both `aligned_part_contour_transformed`
            # and `reference_centered_final` are in the same relative pixel space.
            # To get deviation in mm, we divide by `reference_pixels_per_mm`.

            # Adjust centroids back for deviation calculation (optional, can work with centered)
            # For accurate deviation, we need points in a common real-world coordinate system (mm)
            # and ideally, originating from their 'original' positions before centering.

            # Revert centering for both and convert to mm
            aligned_part_contour_mm = (aligned_part_contour_transformed.reshape(-1, 2) + reference_centroid) / reference_pixels_per_mm
            reference_contour_mm = (reference_contour_flat + reference_centroid) / reference_pixels_per_mm

            # Area Calculation (in mm²)
            image_area_mm2 = cv2.contourArea(aligned_part_contour_mm)
            reference_area_mm2 = cv2.contourArea(reference_contour_mm)

            area_deviation_percent = ((image_area_mm2 - reference_area_mm2) / reference_area_mm2) * 100 if reference_area_mm2 != 0 else 0

            # Calculate Maximum Contour Deviation (in mm)
            # Use KDTree for efficient nearest neighbor search
            if len(aligned_part_contour_mm) > 0 and len(reference_contour_mm) > 0:
                kdtree_reference = KDTree(reference_contour_mm)
                distances, _ = kdtree_reference.query(aligned_part_contour_mm)
                max_deviation_mm = np.max(distances)
            else:
                max_deviation_mm = 0.0
                logger.warning("One of the contours is empty after alignment, cannot calculate max deviation.")

            self.progress_updated_with_text.emit(80, "Generating visual outputs...")

            # --- Visual Outputs ---
            # Create superimposed image
            # The contours are currently float. Convert to int for drawing.
            # Both need to be put on a common canvas, which implies common scale and origin.
            # Let's use the reference_original_image's coordinate system for superposition,
            # or a new blank image based on reference image dimensions.
            
            # Draw the aligned part contour onto the reference image.
            # Need to convert aligned_part_contour_transformed (in reference's pixel space, centered)
            # back to original reference image coordinates for drawing.
            aligned_part_contour_display = (aligned_part_contour_transformed.reshape(-1, 2) + reference_centroid).astype(np.int32)
            
            # Ensure reference_original_image is BGR for drawing colored contours
            if len(reference_original_image.shape) == 2:
                superimposed_image = cv2.cvtColor(reference_original_image, cv2.COLOR_GRAY2BGR)
            else:
                superimposed_image = reference_original_image.copy()

            # Draw the original reference contour (white)
            cv2.drawContours(superimposed_image, [reference_contour], -1, (255, 255, 255), 2)
            # Draw the aligned part contour (green)
            cv2.drawContours(superimposed_image, [aligned_part_contour_display], -1, (0, 255, 0), 2)

            # Prepare processed images for display in UI and report
            part_processed_display_image = image_processor.draw_contour_on_image(part_original_image, part_contour)
            reference_processed_display_image = image_processor.draw_contour_on_image(reference_original_image, reference_contour)
            
            # Save temporary images for reporting
            # Create a temporary directory or use a known temp path
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            part_img_temp_path = os.path.join(temp_dir, f"part_processed_{timestamp}.png")
            reference_img_temp_path = os.path.join(temp_dir, f"reference_processed_{timestamp}.png")
            superimposed_img_temp_path = os.path.join(temp_dir, f"superimposed_{timestamp}.png")

            cv2.imwrite(part_img_temp_path, part_processed_display_image)
            cv2.imwrite(reference_img_temp_path, reference_processed_display_image)
            cv2.imwrite(superimposed_img_temp_path, superimposed_image)


            self.progress_updated_with_text.emit(90, "Finalizing report data...")

            # Prepare results dictionary
            results = {
                'part_image_path': self.config.image_path,
                'reference_path': self.config.reference_path,
                'reference_type': self.config.reference_type,
                'part_pixels_per_mm': part_pixels_per_mm,
                'reference_pixels_per_mm': reference_pixels_per_mm,
                'part_area_mm2': image_area_mm2,
                'reference_area_mm2': reference_area_mm2,
                'area_deviation_percent': area_deviation_percent,
                'max_deviation_mm': max_deviation_mm,
                'aligned_part_contour': aligned_part_contour_display, # Pass aligned part contour for UI drawing
                'aligned_reference_contour': reference_contour, # Pass original reference contour
                
                # These are the NumPy arrays for direct UI display in BoundaryComparisonWindow
                'superimposed_image': superimposed_image,
                'part_processed_display_image': part_processed_display_image,
                'reference_processed_display_image': reference_processed_display_image,
                
                # These are the paths to temporary files for the PDF report
                'processed_part_image_path': part_img_temp_path,
                'processed_reference_image_path': reference_img_temp_path,
                'superimposed_image_path': superimposed_img_temp_path,

                'part_contour': part_contour # Pass original part contour
            }

            self.progress_updated_with_text.emit(100, "Analysis Complete!")
            logger.info("Analysis completed successfully.")
            self.analysis_completed.emit(results)

        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}", exc_info=True)
            self.analysis_error.emit(f"An error occurred: {e}")

    # Helper for pseudo-scale (used if calibration values are missing)
    # This method is now primarily handled by ImageProcessor, but kept here if needed for direct use.
    def _get_pseudo_pixels_per_mm(self, image_shape):
        """
        Calculates a pseudo pixels/mm ratio based on image diagonal,
        assuming a standard "known" diagonal length in mm (e.g., 100mm).
        This is a fallback for when explicit calibration is not provided.
        """
        height, width = image_shape[:2]
        diagonal_pixels = (width**2 + height**2)**0.5
        # Assume a standard diagonal of 100mm for pseudo-scaling
        pseudo_known_distance_mm = 100.0
        if diagonal_pixels == 0:
            logger.warning("Image has zero diagonal pixels, returning default pseudo pixels/mm.")
            return 1.0 # Default fallback

        pseudo_pixels_per_mm = diagonal_pixels / pseudo_known_distance_mm
        logger.info(f"Calculated pseudo pixels/mm: {pseudo_pixels_per_mm:.2f}")
        return pseudo_pixels_per_mm