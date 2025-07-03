# src/image_processing.py
import cv2
import numpy as np
import trimesh
from scipy.spatial import KDTree, ConvexHull # Added ConvexHull import
from scipy.ndimage import rotate as ndimage_rotate

from .utils import logger

# Helper function for consistent angle from minAreaRect
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

class ImageProcessor:
    """
    Handles loading, processing, and contour extraction for images (part and reference).
    Also manages STL projection if the reference is an STL model.
    """
    def __init__(self, config):
        self.config = config
        # Access dataclass attributes directly, not with .get()
        self.image_path = config.image_path
        self.reference_path = config.reference_path
        self.reference_type = config.reference_type
        self.blur_kernel = config.blur_kernel # Corrected to blur_kernel
        self.processing_method = config.processing_method
        self.canny_threshold1 = config.canny_threshold1
        self.canny_threshold2 = config.canny_threshold2
        self.threshold_value = config.threshold_value
        self.projection_axis = config.projection_axis
        self.part_pixels_per_mm = config.part_pixels_per_mm_value
        self.reference_pixels_per_mm = config.reference_pixels_per_mm_value

    def load_and_process_image(self, image_path):
        """
        Loads an image, converts it to grayscale, applies blur, and extracts contour
        based on the configured processing method.
        Returns original image, grayscale, edge/thresholded, and largest contour.
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            raise FileNotFoundError(f"Image not found at {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0) # Use self.blur_kernel

        if self.processing_method == 'Canny':
            edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else: # Threshold
            _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)
            
            # Invert binary image if the object is darker than background (common for parts)
            # A simple heuristic: if the average pixel value is high, assume light background, dark object.
            # Otherwise, assume dark background, light object.
            if np.mean(binary) > 127: # Mostly white, likely background
                binary = cv2.bitwise_not(binary) # Invert
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning(f"No contours found in image: {image_path}. Returning empty contour.")
            return img, gray, edges if self.processing_method == 'Canny' else binary, np.array([]) # Return empty numpy array for contour

        # Find the largest contour, assumed to be the part outline
        largest_contour = max(contours, key=cv2.contourArea)
        logger.info(f"Largest contour found with area: {cv2.contourArea(largest_contour)}")
        return img, gray, edges if self.processing_method == 'Canny' else binary, largest_contour

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
        logger.warning(f"Using pseudo pixels/mm (fallback): {pseudo_pixels_per_mm:.2f}")
        return pseudo_pixels_per_mm

    def process_part_image(self):
        """Processes the part image and returns its contour and original image."""
        original_image, gray_image, edge_image, part_contour = self.load_and_process_image(self.image_path)
        if self.part_pixels_per_mm is None:
            self.part_pixels_per_mm = self._get_pseudo_pixels_per_mm(original_image.shape)
            logger.warning(f"Part image pixels/mm not explicitly set. Using pseudo value: {self.part_pixels_per_mm:.2f}")

        return original_image, part_contour, self.part_pixels_per_mm

    def process_reference(self):
        """Processes the reference (image or STL) and returns its contour and pixels/mm."""
        if self.reference_type == 'image':
            original_image, gray_image, edge_image, reference_contour = self.load_and_process_image(self.reference_path)
            if self.reference_pixels_per_mm is None:
                self.reference_pixels_per_mm = self._get_pseudo_pixels_per_mm(original_image.shape)
                logger.warning(f"Reference image pixels/mm not explicitly set. Using pseudo value: {self.reference_pixels_per_mm:.2f}")
            
            return original_image, reference_contour, self.reference_pixels_per_mm
        
        elif self.reference_type == 'stl':
            self.logger.info(f"Processing STL file: {self.reference_path}")
            try:
                mesh = trimesh.load_mesh(self.reference_path)
                
                # If the mesh has multiple bodies, combine them or choose the largest
                # For simplicity, we'll just use the main mesh and its vertices/faces
                
                # Get 2D projection of the mesh based on the specified axis
                # trimesh.projections.project_polygon_on_plane gives a 2D polygon
                # Then convert this to an OpenCV contour
                
                # Determine projection plane based on axis
                if self.projection_axis == 'X':
                    # Project onto YZ plane, so X coordinate is ignored
                    projection_vertices = mesh.vertices[:, [1, 2]] 
                elif self.projection_axis == 'Y':
                    # Project onto XZ plane, so Y coordinate is ignored
                    projection_vertices = mesh.vertices[:, [0, 2]]
                else: # Default to Z or if 'Z' is specified
                    # Project onto XY plane, so Z coordinate is ignored
                    projection_vertices = mesh.vertices[:, [0, 1]]

                # Calculate the Convex Hull of the projected points to get the outer boundary
                # This ensures we get a single, closed contour for comparison
                if len(projection_vertices) < 3:
                    raise ValueError("Not enough unique points in STL for projection.")
                
                hull = ConvexHull(projection_vertices) # Use ConvexHull from scipy.spatial
                # The vertices of the convex hull are the boundary points
                stl_contour = projection_vertices[hull.vertices]

                # Scale the STL contour to pixel space for comparison with the image
                # Need to determine a pixels/mm ratio for the STL if not provided
                if self.reference_pixels_per_mm is None:
                    # Calculate min/max bounds of the projected contour in mm
                    min_coords = np.min(stl_contour, axis=0)
                    max_coords = np.max(stl_contour, axis=0)
                    width_mm = max_coords[0] - min_coords[0]
                    height_mm = max_coords[1] - min_coords[1]
                    
                    if width_mm == 0 or height_mm == 0:
                        logger.warning("Projected STL has zero width or height, cannot calculate pseudo pixels/mm.")
                        # Fallback to a default pseudo scale
                        stl_pixels_per_mm = 10.0 # Arbitrary default
                    else:
                        # Assume a square pixel aspect ratio.
                        # We need a reference "pixel size" for the STL projection.
                        # Let's set a default virtual resolution for STL projection.
                        # For example, assume we want the larger dimension of the STL to be 500 pixels.
                        max_dim_mm = max(width_mm, height_mm)
                        if max_dim_mm == 0:
                             stl_pixels_per_mm = 1.0 # Prevent division by zero
                        else:
                            target_max_pixels = 500.0 # Arbitrary target pixel size for the STL rendering
                            stl_pixels_per_mm = target_max_pixels / max_dim_mm
                    
                    self.reference_pixels_per_mm = stl_pixels_per_mm
                    logger.warning(f"Reference STL pixels/mm not explicitly set. Using pseudo value: {self.reference_pixels_per_mm:.2f}")
                else:
                    stl_pixels_per_mm = self.reference_pixels_per_mm
                
                # Translate points so the smallest coordinate is at (0,0) and then scale
                # The contour needs to be integers for OpenCV
                scaled_translated_contour = (stl_contour - np.min(stl_contour, axis=0)) * stl_pixels_per_mm
                stl_contour_pixels = scaled_translated_contour.astype(np.int32)
                
                # OpenCV contours expect (N, 1, 2) shape, so reshape
                stl_contour_pixels = stl_contour_pixels.reshape(-1, 1, 2)

                # Create a blank image to draw the STL contour onto
                # The size of this image should be based on the max dimensions of the scaled contour
                # Add some padding
                max_x = np.max(stl_contour_pixels[:,:,0])
                max_y = np.max(stl_contour_pixels[:,:,1])
                img_width = int(max_x + 10)
                img_height = int(max_y + 10)
                
                # Ensure minimum image size
                img_width = max(img_width, 1)
                img_height = max(img_height, 1)

                blank_image = np.zeros((img_height, img_width, 3), dtype=np.uint8) # Black background
                cv2.drawContours(blank_image, [stl_contour_pixels], -1, (255, 255, 255), 2) # White contour

                logger.info(f"STL projected contour generated from {self.projection_axis}-axis view.")
                
                # For consistency, return the blank image with contour drawn as "original_image"
                return blank_image, stl_contour_pixels, stl_pixels_per_mm

            except Exception as e:
                logger.error(f"Error processing STL file {self.reference_path}: {e}", exc_info=True)
                raise
    
    def draw_contour_on_image(self, image, contour, color=(0, 255, 0), thickness=2):
        """Draws a contour on a copy of the given image and returns it."""
        if image is None or contour is None:
            logger.warning("Attempted to draw contour on None image or with None contour.")
            return image
        
        display_image = image.copy()
        if len(display_image.shape) == 2: # Convert grayscale to BGR for consistent coloring
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(display_image, [contour], -1, color, thickness)
        return display_image