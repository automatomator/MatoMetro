# src/utils.py

import numpy as np
import cv2
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QSize
import logging

def setup_logger():
    """Sets up a basic logger."""
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger(__name__)

logger = setup_logger()

def convert_np_to_qpixmap(np_image, target_size: QSize = None):
    """
    Converts a NumPy array (assumed grayscale or BGR) to QPixmap,
    optionally scaling it to a target QSize while maintaining aspect ratio.
    """
    if np_image is None or np_image.size == 0:
        logger.warning("Attempted to convert an empty or None numpy image to QPixmap.")
        return QPixmap()

    # Ensure the image is 8-bit for QImage
    if np_image.dtype != np.uint8:
        if np_image.dtype in [np.float32, np.float64]:
            if np.max(np_image) <= 1.0 + np.finfo(float).eps:
                np_image = (np_image * 255).astype(np.uint8)
            else:
                np_image = np_image.astype(np.uint8)
        elif np_image.dtype == bool:
            np_image = (np_image * 255).astype(np.uint8)
        else:
            logger.error(f"Unsupported numpy image dtype: {np_image.dtype}")
            return QPixmap()

    height, width = np_image.shape[:2]
    bytes_per_line = width * (3 if len(np_image.shape) > 2 else 1)

    if len(np_image.shape) == 3:
        q_img = QImage(np_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
    else:
        q_img = QImage(np_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)

    pixmap = QPixmap.fromImage(q_img)

    if target_size and not target_size.isEmpty():
        pixmap = pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    return pixmap