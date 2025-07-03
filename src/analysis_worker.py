# src/analysis_worker.py
import numpy as np
import trimesh
import cv2
from scipy.spatial import ConvexHull, KDTree
from scipy.ndimage import rotate as ndimage_rotate
from PyQt6.QtCore import QThread, pyqtSignal
import tempfile 
import os # Import os

from .utils import logger
from .image_processing import ImageProcessor # Import the ImageProcessor

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
        # Access dataclass attributes directly, not with .get()
        self.image_path = config.image_path
        self.reference_path = config.reference_path
        self.reference_type = config.reference_type
        self.projection_axis = config.projection_axis
        self.blur_kernel = config.blur_kernel
        self.processing_method = config.processing_method
        self.canny_threshold1 = config.canny_threshold1
        self.canny_threshold2 = config.canny_threshold2
        self.threshold_value = config.threshold_value
        self.part_pixels_per_mm_value = config.part_pixels_per_mm_value
        self.reference_pixels_per_mm_value = config.reference_pixels_per_mm_value

    def run(self):
        try:
            self.progress_updated_with_text.emit(5, "Loading Images")
            
            # Initialize ImageProcessor with the config object
            image_processor = ImageProcessor(self.config)

            # Process part image
            original_part_image, part_contour, part_pixels_per_mm = image_processor.process_part_image()
            self.progress_updated_with_text.emit(20, "Processing Part Image")

            # Process reference (image or STL)
            original_reference_image, reference_contour, reference_pixels_per_mm = image_processor.process_reference()
            self.progress_updated_with_text.emit(40, "Processing Reference File")

            # --- Alignment ---
            self.progress_updated_with_text.emit(60, "Aligning Contours")

            # Calculate initial centroids
            M_part = cv2.moments(part_contour)
            if M_part["m00"] == 0:
                raise ValueError("Part contour has zero area, cannot proceed with analysis.")
            cx_part = int(M_part["m10"] / M_part["m00"])
            cy_part = int(M_part["m01"] / M_part["m00"])

            M_ref = cv2.moments(reference_contour)
            if M_ref["m00"] == 0:
                raise ValueError("Reference contour has zero area, cannot proceed with analysis.")
            cx_ref = int(M_ref["m10"] / M_ref["m00"])
            cy_ref = int(M_ref["m01"] / M_ref["m00"])

            # Translate contours so their centroids align
            translation_x = cx_part - cx_ref
            translation_y = cy_part - cy_ref
            
            # Apply initial translation to reference contour
            translated_reference_contour = reference_contour + np.array([translation_x, translation_y])
            
            # Convert contours to float32 and reshape to (N, 2) for estimateAffine2D
            part_contour_float = part_contour.reshape(-1, 2).astype(np.float32) # Optimized
            translated_reference_contour_float = translated_reference_contour.reshape(-1, 2).astype(np.float32) # Optimized

            # Attempt robust alignment using estimateAffine2D
            # This requires at least 3 points in each contour.
            M = None
            if len(part_contour_float) >= 3 and len(translated_reference_contour_float) >= 3:
                try:
                    # Use RANSAC for robustness against outliers
                    M, inliers = cv2.estimateAffine2D(translated_reference_contour_float, part_contour_float,
                                                    method=cv2.RANSAC, ransacReprojThreshold=5.0)
                    logger.info("Contours aligned using estimateAffine2D (RANSAC).")
                except cv2.error as e:
                    logger.error(f"Error during contour alignment: {e}")
                    logger.warning("Falling back to simple centroid alignment.")
                    # Fallback to simple centroid alignment if robust alignment fails
                    M = None # Indicate that robust alignment failed
            else:
                logger.warning("Not enough points in contours for robust affine alignment. Falling back to simple centroid alignment.")
                M = None # Indicate that robust alignment skipped

            aligned_reference_contour = None
            if M is not None:
                # Apply the affine transformation to the original reference contour
                # Need to reshape reference_contour back to (N, 1, 2) for cv2.transform
                aligned_reference_contour = cv2.transform(reference_contour.astype(np.float32), M).astype(np.int32)
                aligned_reference_contour = aligned_reference_contour.reshape(-1, 1, 2) # Ensure (N, 1, 2) shape
            else:
                # If estimateAffine2D failed or skipped, just use the centroid-aligned contour
                aligned_reference_contour = translated_reference_contour.astype(np.int32).reshape(-1, 1, 2)
                logger.info("Using simple centroid alignment for reference contour.")

            # --- Calculate Deviations ---
            self.progress_updated_with_text.emit(70, "Calculating Deviations")

            # Calculate contour areas in pixels
            part_area_pixels = cv2.contourArea(part_contour)
            reference_area_pixels = cv2.contourArea(aligned_reference_contour) # Use aligned reference contour

            # Convert areas to mm²
            # Area in mm² = Area in pixels² / (pixels/mm)²
            part_area_mm2 = part_area_pixels / (part_pixels_per_mm**2)
            reference_area_mm2 = reference_area_pixels / (reference_pixels_per_mm**2)

            # Area Deviation
            area_deviation_percent = ((part_area_mm2 - reference_area_mm2) / reference_area_mm2) * 100 if reference_area_mm2 != 0 else 0

            # Calculate maximum point-to-point deviation
            # Use KDTree for efficient nearest neighbor search
            kdtree_ref = KDTree(aligned_reference_contour.reshape(-1, 2))
            distances, _ = kdtree_ref.query(part_contour.reshape(-1, 2))
            max_deviation_pixels = np.max(distances)
            max_deviation_mm = max_deviation_pixels / part_pixels_per_mm # Convert to mm

            # --- Generate Visual Outputs ---
            self.progress_updated_with_text.emit(80, "Generating Visual Outputs")

            # Create a combined image for visualization
            # Determine the size of the combined image
            all_contours = np.vstack([part_contour.reshape(-1, 2), aligned_reference_contour.reshape(-1, 2)])
            min_x, min_y = np.min(all_contours, axis=0)
            max_x, max_y = np.max(all_contours, axis=0)

            # Add padding to the boundaries
            padding = 50
            display_width = int(max_x - min_x + 2 * padding)
            display_height = int(max_y - min_y + 2 * padding)

            # Create blank images for drawing
            # Ensure the blank image is BGR for color drawing
            part_boundary_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            reference_boundary_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            superimposed_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)

            # Offset contours to fit into the new blank images
            offset_x = -min_x + padding
            offset_y = -min_y + padding
            
            part_contour_offset = (part_contour + np.array([offset_x, offset_y])).astype(np.int32)
            aligned_reference_contour_offset = (aligned_reference_contour + np.array([offset_x, offset_y])).astype(np.int32)

            # Draw contours
            cv2.drawContours(part_boundary_image, [part_contour_offset], -1, (0, 255, 0), 2) # Green for part
            cv2.drawContours(reference_boundary_image, [aligned_reference_contour_offset], -1, (255, 0, 0), 2) # Blue for reference
            
            # Superimposed: part (green) and aligned reference (blue)
            superimposed_image = cv2.addWeighted(part_boundary_image, 0.5, reference_boundary_image, 0.5, 0)
            
            # Generate temporary file paths for saving images
            temp_dir = tempfile.gettempdir()
            part_boundary_path = os.path.join(temp_dir, "part_boundary.png")
            reference_boundary_path = os.path.join(temp_dir, "reference_boundary.png")
            superimposed_path = os.path.join(temp_dir, "superimposed_result.png")

            cv2.imwrite(part_boundary_path, part_boundary_image)
            cv2.imwrite(reference_boundary_path, reference_boundary_image)
            cv2.imwrite(superimposed_path, superimposed_image)
            logger.info(f"Superimposed image saved to {superimposed_path}")

            results = {
                'image_path': self.image_path,
                'reference_path': self.reference_path,
                'part_area_mm2': part_area_mm2,
                'reference_area_mm2': reference_area_mm2,
                'area_deviation_percent': area_deviation_percent,
                'max_deviation_mm': max_deviation_mm,
                'part_boundary_image_path': part_boundary_path,
                'reference_boundary_image_path': reference_boundary_path,
                'superimposed_image_path': superimposed_path,
                'part_contour': part_contour # Pass original part contour
            }

            self.progress_updated_with_text.emit(100, "Analysis Complete!")
            logger.info("Analysis completed successfully.")
            self.analysis_completed.emit(results)

        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}", exc_info=True)
            self.analysis_error.emit(f"An error occurred: {e}")