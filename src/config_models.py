# src/config_models.py

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal

@dataclass
class AnalysisConfig:
    """
    Configuration for the Part Analysis.

    Attributes:
        image_path (str): Path to the image of the physical part.
        reference_path (str): Path to the reference (image or STL) file.
        reference_type (Literal['image', 'stl']): Type of the reference file.
        projection_axis (Literal['X', 'Y', 'Z']): For STL references, the axis along which to project.
        threshold_value (int): Threshold value for binary thresholding (0-255).
        canny_threshold1 (int): First threshold for the Canny edge detector.
        canny_threshold2 (int): Second threshold for the Canny edge detector.
        blur_kernel (int): Kernel size for Gaussian blur. Must be odd.
        processing_method (Literal['Canny', 'Threshold']): Method for contour extraction.
        part_pixels_per_mm_value (Optional[float]): Calibration for the part image (pixels per mm).
        reference_pixels_per_mm_value (Optional[float]): Calibration for the reference image (pixels per mm).
        # Add other configuration parameters as needed
    """
    image_path: str
    reference_path: str
    reference_type: Literal['image', 'stl'] = 'image'
    projection_axis: Literal['X', 'Y', 'Z'] = 'Z' # Default to Z (top view for most CAD)
    threshold_value: int = 127
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    blur_kernel: int = 5 # Ensure this is always odd in usage or add validation
    processing_method: Literal['Canny', 'Threshold'] = 'Canny'
    part_pixels_per_mm_value: Optional[float] = None
    reference_pixels_per_mm_value: Optional[float] = None

    def __post_init__(self):
        # Basic validation for blur_kernel to ensure it's odd
        if self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be an odd number.")

    def to_dict(self):
        """Converts the dataclass instance to a dictionary."""
        return asdict(self)