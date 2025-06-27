import numpy as np
import cv2
import sys
import os

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSizePolicy, QLineEdit,
    QProgressBar, QGridLayout, QGroupBox, QRadioButton,
    QScrollArea, QDialog,
)
from PyQt6.QtGui import QPixmap, QImage, QDoubleValidator, QIntValidator, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QPoint, QLineF

# Import AnalysisWorker from your new module
from .analysis_worker import AnalysisWorker
# Import utils
from .utils import convert_np_to_qpixmap, logger # Import logger from utils

# --- Boundary Comparison Window ---
# This window remains largely the same, as it just displays results from the worker
class BoundaryComparisonWindow(QWidget):
    """
    A separate window to display the image with detected boundary,
    STL projection, and a superimposed view.
    """
    def __init__(self, results, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Boundary Comparison")
        self.setFixedSize(900, 650) # Adjusted size for better layout

        self.results = results
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        # Labels for displaying images
        self.image_boundary_label = QLabel("Part Boundary")
        self.image_boundary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_boundary_label.setFixedSize(250, 250) # Fixed size for consistency
        self.image_boundary_label.setScaledContents(True)

        self.stl_projection_label = QLabel("Reference Boundary")
        self.stl_projection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stl_projection_label.setFixedSize(250, 250)
        self.stl_projection_label.setScaledContents(True)

        self.superimposed_label = QLabel("Superimposed (Aligned)")
        self.superimposed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.superimposed_label.setFixedSize(250, 250)
        self.superimposed_label.setScaledContents(True)

        # Add labels to horizontal layout
        image_layout.addWidget(self.image_boundary_label)
        image_layout.addWidget(self.stl_projection_label)
        image_layout.addWidget(self.superimposed_label)

        main_layout.addLayout(image_layout)

        # Results display
        self.results_text_label = QLabel("Analysis Results:")
        self.results_display = QLabel()
        self.results_display.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.results_display.setWordWrap(True)

        # Add a scroll area for results_display if text is long
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.results_display)
        scroll_area.setFixedHeight(150) # Limit height of scroll area

        main_layout.addWidget(self.results_text_label)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        self.display_results()

    def display_results(self):
        # Convert numpy arrays to QPixmap and display
        self.image_boundary_label.setPixmap(convert_np_to_qpixmap(self.results['processed_image'], self.image_boundary_label.size()))
        self.stl_projection_label.setPixmap(convert_np_to_qpixmap(self.results['projected_stl_image'], self.stl_projection_label.size()))
        self.superimposed_label.setPixmap(convert_np_to_qpixmap(self.results['superimposed_image'], self.superimposed_label.size()))

        # For area_deviation_percent and max_deviation_mm, check if they exist in results
        # These are crucial for the output, ensure they are computed by AnalysisWorker
        image_area_mm2 = self.results.get('image_area_mm2', 0.0)
        reference_area_mm2 = self.results.get('reference_area_mm2', 0.0)
        area_deviation_percent = self.results.get('area_deviation_percent', 0.0)
        max_deviation_mm = self.results.get('max_deviation_mm', 0.0)

        results_str = (
            f"Image Area: {image_area_mm2:.2f} mm²\n"
            f"Reference Area: {reference_area_mm2:.2f} mm²\n"
            f"Area Deviation: {area_deviation_percent:.2f} %\n"
            f"Maximum Contour Deviation: {max_deviation_mm:.2f} mm"
        )
        self.results_display.setText(results_str)


# --- Main Application Window ---
class PartAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Part Deviation Analyzer")
        self.setFixedSize(800, 650) # Adjusted size for new controls and removed calibration

        self.image_path = None
        self.reference_path = None
        self.reference_type = None # 'image' or 'stl'
        self.projection_axis = 'Z' # Default (if STL is re-enabled)

        self.analysis_worker = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout() # Layout for file selection and processing params

        # --- File Selection Group ---
        file_selection_group = QGroupBox("File Selection")
        file_selection_layout = QVBoxLayout()

        # Image Selection
        image_layout = QHBoxLayout()
        self.select_image_button = QPushButton("Select Part Image")
        self.select_image_button.clicked.connect(self.select_image)
        self.image_path_label = QLabel("No Image Selected")
        image_layout.addWidget(self.select_image_button)
        image_layout.addWidget(self.image_path_label)
        file_selection_layout.addLayout(image_layout)

        # Reference Selection
        reference_layout = QHBoxLayout()
        self.select_reference_button = QPushButton("Select Reference")
        self.select_reference_button.clicked.connect(self.select_reference)
        self.reference_path_label = QLabel("No Reference Model/Image Selected")
        reference_layout.addWidget(self.select_reference_button)
        reference_layout.addWidget(self.reference_path_label)
        file_selection_layout.addLayout(reference_layout)

        file_selection_group.setLayout(file_selection_layout)
        top_layout.addWidget(file_selection_group)

        # --- Image Processing Parameters Group ---
        processing_group = QGroupBox("Image Processing Parameters")
        processing_layout = QGridLayout()

        # Processing Method Radio Buttons
        self.radio_canny = QRadioButton("Canny Edge Detection")
        self.radio_threshold = QRadioButton("Binary Threshold")
        self.radio_canny.setChecked(True) # Default to Canny
        self.radio_canny.toggled.connect(self.update_processing_inputs_visibility)
        self.radio_threshold.toggled.connect(self.update_processing_inputs_visibility)
        processing_layout.addWidget(self.radio_canny, 0, 0, 1, 2)
        processing_layout.addWidget(self.radio_threshold, 1, 0, 1, 2)

        # Canny Thresholds
        self.canny_threshold1_label = QLabel("Canny Threshold 1:")
        self.canny_threshold1_input = QLineEdit("50")
        self.canny_threshold1_input.setValidator(QIntValidator(0, 255))
        self.canny_threshold2_label = QLabel("Canny Threshold 2:")
        self.canny_threshold2_input = QLineEdit("150")
        self.canny_threshold2_input.setValidator(QIntValidator(0, 255))
        processing_layout.addWidget(self.canny_threshold1_label, 2, 0)
        processing_layout.addWidget(self.canny_threshold1_input, 2, 1)
        processing_layout.addWidget(self.canny_threshold2_label, 3, 0)
        processing_layout.addWidget(self.canny_threshold2_input, 3, 1)

        # Blur Kernel Size
        self.blur_kernel_label = QLabel("Blur Kernel Size (Odd):")
        self.blur_kernel_input = QLineEdit("5")
        self.blur_kernel_input.setValidator(QIntValidator(0, 21)) # Allow 0 for no blur
        self.blur_kernel_input.textChanged.connect(self.validate_odd_kernel) # NEW: Validation
        processing_layout.addWidget(self.blur_kernel_label, 4, 0)
        processing_layout.addWidget(self.blur_kernel_input, 4, 1)

        # Threshold Value
        self.threshold_value_label = QLabel("Threshold Value (0-255):")
        self.threshold_value_input = QLineEdit("127")
        self.threshold_value_input.setValidator(QIntValidator(0, 255))
        processing_layout.addWidget(self.threshold_value_label, 5, 0)
        processing_layout.addWidget(self.threshold_value_input, 5, 1)

        processing_group.setLayout(processing_layout)
        top_layout.addWidget(processing_group)
        main_layout.addLayout(top_layout)

        # Removed Calibration Group

        # --- Projection Axis Group (currently disabled as only image reference supported) ---
        self.projection_group = QGroupBox("STL Projection Axis (currently not used)")
        projection_layout = QHBoxLayout()
        self.radio_x = QRadioButton("X-axis")
        self.radio_y = QRadioButton("Y-axis")
        self.radio_z = QRadioButton("Z-axis")
        self.radio_z.setChecked(True) # Default
        self.radio_x.toggled.connect(lambda: self.set_projection_axis('X'))
        self.radio_y.toggled.connect(lambda: self.set_projection_axis('Y'))
        self.radio_z.toggled.connect(lambda: self.set_projection_axis('Z'))
        projection_layout.addWidget(self.radio_x)
        projection_layout.addWidget(self.radio_y)
        projection_layout.addWidget(self.radio_z)
        self.projection_group.setLayout(projection_layout)
        self.projection_group.setEnabled(False) # Always disabled for now as STL is not supported
        main_layout.addWidget(self.projection_group)

        # --- Progress Bar and Labels ---
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Progress: %p%")
        self.progress_bar.setVisible(False) # Hidden initially
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False) # Hidden initially

        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.progress_label)

        # --- Action Buttons ---
        button_layout = QHBoxLayout()
        self.perform_analysis_button = QPushButton("Perform Analysis")
        self.perform_analysis_button.clicked.connect(self.start_analysis)
        self.perform_analysis_button.setEnabled(False) # Disabled until files selected

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_application_state)

        button_layout.addWidget(self.perform_analysis_button)
        button_layout.addWidget(self.reset_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.update_processing_inputs_visibility() # Initial call to set visibility
        self.check_analysis_button_state() # Initial check for analysis button state


    def validate_odd_kernel(self, text):
        """Ensures the blur kernel size is an odd integer or 0."""
        if text == "0": # Allow 0 for no blur
            return
        if text.isdigit():
            val = int(text)
            if val % 2 == 0:
                QMessageBox.warning(self, "Invalid Input", "Blur kernel size must be an odd number (or 0 for no blur). Correcting to nearest odd.")
                # Auto-correct to nearest odd (e.g., 4 -> 5, 6 -> 7)
                self.blur_kernel_input.setText(str(val + 1))
        elif text != "": # Only warn if not empty string (user might be typing)
            QMessageBox.warning(self, "Invalid Input", "Blur kernel size must be a positive integer or 0.")
            self.blur_kernel_input.setText("5") # Reset to default if non-digit


    def update_processing_inputs_visibility(self):
        """Hides/shows relevant inputs based on selected processing method."""
        is_canny = self.radio_canny.isChecked()
        self.canny_threshold1_label.setVisible(is_canny)
        self.canny_threshold1_input.setVisible(is_canny)
        self.canny_threshold2_label.setVisible(is_canny)
        self.canny_threshold2_input.setVisible(is_canny)

        is_threshold = self.radio_threshold.isChecked()
        self.threshold_value_label.setVisible(is_threshold)
        self.threshold_value_input.setVisible(is_threshold)


    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Part Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            self.image_path_label.setText(os.path.basename(file_path))
            logger.info(f"Part image selected: {file_path}")
            self.check_analysis_button_state()


    def select_reference(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference Model or Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;STL Files (*.stl)")
        if file_path:
            self.reference_path = file_path
            self.reference_path_label.setText(os.path.basename(file_path))
            # Current implementation in analysis_worker only supports 'image' for direct contour alignment
            # The STL part would need a separate projection logic.
            # For now, we force to image type, and disable STL axis selection
            if file_path.lower().endswith('.stl'):
                QMessageBox.information(self, "STL Support", "STL reference is not fully implemented for direct contour alignment yet. Please select an image reference.")
                self.reference_path = None # Clear selection if it's an unsupported STL
                self.reference_path_label.setText("No Reference Model/Image Selected")
                self.reference_type = None
            else:
                self.reference_type = 'image'
                logger.info(f"Image reference selected: {file_path}")
            self.check_analysis_button_state()


    def set_projection_axis(self, axis):
        self.projection_axis = axis
        logger.info(f"Projection axis set to: {self.projection_axis}")

    def check_analysis_button_state(self):
        """
        Enables/disables the 'Perform Analysis' button based on whether
        both image and reference files are selected.
        """
        image_selected = self.image_path is not None
        reference_selected = self.reference_path is not None
        
        self.perform_analysis_button.setEnabled(image_selected and reference_selected)


    def start_analysis(self):
        if not self.image_path or not self.reference_path:
            QMessageBox.warning(self, "Missing Files", "Please select both a part image and a reference file.")
            return

        try:
            # Collect all configuration parameters
            config = {
                'image_path': self.image_path,
                'reference_path': self.reference_path,
                'reference_type': self.reference_type,
                'projection_axis': self.projection_axis, # Will only be used if STL is re-enabled later
                # New parameters from UI inputs
                'threshold_value': int(self.threshold_value_input.text()),
                'canny_threshold1': int(self.canny_threshold1_input.text()),
                'canny_threshold2': int(self.canny_threshold2_input.text()),
                'blur_kernel': int(self.blur_kernel_input.text()),
                'processing_method': 'Canny' if self.radio_canny.isChecked() else 'Threshold'
            }

            self.perform_analysis_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)

            logger.info("Starting analysis worker with configuration:")
            for k, v in config.items():
                logger.info(f"  {k}: {v}")

            self.analysis_worker = AnalysisWorker(config) # Pass the entire config dict
            self.analysis_worker.progress_updated_with_text.connect(self.update_progress)
            self.analysis_worker.analysis_completed.connect(self.on_analysis_complete)
            self.analysis_worker.analysis_error.connect(self.on_analysis_error)
            self.analysis_worker.start()

        except ValueError as e:
            QMessageBox.critical(self, "Input Error", str(e))
            self.reset_ui_state() # Reset UI on input error
        except Exception as e:
            logger.error(f"Unexpected error starting analysis: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")
            self.reset_ui_state()


    def update_progress(self, value, text):
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)

    def on_analysis_complete(self, results):
        logger.info("Analysis completed successfully. Displaying results.")
        self.reset_ui_state() # Reset UI elements like progress bar
        
        self.boundary_comparison_window = BoundaryComparisonWindow(results, self)
        self.boundary_comparison_window.show()
        
    def on_analysis_error(self, error_message):
        logger.error(f"Analysis failed with error: {error_message}")
        self.reset_ui_state() # Reset UI elements like progress bar
        QMessageBox.critical(self, "Analysis Error", error_message)

    def reset_ui_state(self):
        """Resets the UI elements to their initial state after analysis (or error)."""
        self.perform_analysis_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.progress_label.setText("")

    def reset_application_state(self):
        self.image_path = None
        self.reference_path = None
        self.reference_type = None

        # Reset UI labels
        self.image_path_label.setText("No Image Selected")
        self.reference_path_label.setText("No Reference Model/Image Selected")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Progress: %p%")
        
        # Disable relevant buttons and groups
        self.perform_analysis_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.projection_group.setEnabled(False)

        # Reset image processing parameters to defaults
        self.canny_threshold1_input.setText("50")
        self.canny_threshold2_input.setText("150")
        self.blur_kernel_input.setText("5")
        self.threshold_value_input.setText("127") # Default threshold value

        self.radio_canny.setChecked(True) # Set Canny as default
        self.update_processing_inputs_visibility() # Update visibility on reset

        # If worker is running, stop it
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.quit()
            self.analysis_worker.wait()
            QMessageBox.information(self, "Reset", "Analysis worker stopped.")

        logger.info("Application state reset.")