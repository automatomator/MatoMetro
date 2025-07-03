# src/ui_windows.py
import numpy as np
import cv2
import sys
import os
import tempfile # For temporary files
from datetime import datetime # Import datetime for timestamping report files
from scipy.spatial import ConvexHull # For convex hull calculations

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSizePolicy, QLineEdit,
    QProgressBar, QGridLayout, QGroupBox, QRadioButton,
    QScrollArea, QDialog, QFormLayout, QInputDialog # Added QInputDialog for set_scale method
)
from PyQt6.QtGui import QPixmap, QImage, QDoubleValidator, QIntValidator, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QPoint, QLineF

# Import AnalysisWorker from your new module
from .analysis_worker import AnalysisWorker
# Import utils
from .utils import convert_np_to_qpixmap, logger # Import logger from utils
# Import ReportGenerator
from .report_generator import ReportGenerator
from .config_models import AnalysisConfig # Import the AnalysisConfig dataclass

# --- Boundary Comparison Window ---
# This window remains largely the same, as it just displays results from the worker
class BoundaryComparisonWindow(QWidget):
    """
    A separate window to display the image with detected boundary,
    STL projection, and a superimposed view.
    """
    def __init__(self, results, config, parent=None): # Added config parameter
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Boundary Comparison")
        self.setFixedSize(900, 650) # Adjusted size for better layout
        self.results = results
        self.config = config # Store config for report generation
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.image_display_layout = QGridLayout()

        # Labels for images
        self.part_label = QLabel("Processed Part Boundary:")
        self.part_image_display = QLabel()
        self.part_image_display.setFixedSize(250, 250)
        self.part_image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.part_image_display.setStyleSheet("border: 1px solid gray;")

        self.reference_label = QLabel("Processed/Projected Reference Boundary:")
        self.reference_image_display = QLabel()
        self.reference_image_display.setFixedSize(250, 250)
        self.reference_image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reference_image_display.setStyleSheet("border: 1px solid gray;")

        self.superimposed_label = QLabel("Superimposed (Aligned):")
        self.superimposed_image_display = QLabel()
        self.superimposed_image_display.setFixedSize(250, 250)
        self.superimposed_image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.superimposed_image_display.setStyleSheet("border: 1px solid gray;")

        # Add labels and image displays to grid
        self.image_display_layout.addWidget(self.part_label, 0, 0, Qt.AlignmentFlag.AlignHCenter)
        self.image_display_layout.addWidget(self.reference_label, 0, 1, Qt.AlignmentFlag.AlignHCenter)
        self.image_display_layout.addWidget(self.superimposed_label, 0, 2, Qt.AlignmentFlag.AlignHCenter)
        
        self.image_display_layout.addWidget(self.part_image_display, 1, 0)
        self.image_display_layout.addWidget(self.reference_image_display, 1, 1)
        self.image_display_layout.addWidget(self.superimposed_image_display, 1, 2)
        
        main_layout.addLayout(self.image_display_layout)

        # Quantitative Metrics Display
        metrics_group = QGroupBox("Quantitative Metrics")
        metrics_layout = QFormLayout()

        self.part_area_label = QLabel()
        self.reference_area_label = QLabel()
        self.area_deviation_label = QLabel()
        self.max_deviation_label = QLabel()

        metrics_layout.addRow("Part Area:", self.part_area_label)
        metrics_layout.addRow("Reference Area:", self.reference_area_label)
        metrics_layout.addRow("Area Deviation:", self.area_deviation_label)
        metrics_layout.addRow("Maximum Contour Deviation:", self.max_deviation_label)
        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)

        # Report Generation Button
        self.generate_report_button = QPushButton("Generate PDF Report")
        self.generate_report_button.clicked.connect(self.generate_report)
        main_layout.addWidget(self.generate_report_button)

        self.load_results()
        self.setLayout(main_layout)

    def load_results(self):
        # Display images
        target_size = QSize(self.part_image_display.width(), self.part_image_display.height())

        # Part Boundary Image (from AnalysisWorker's output - already drawn)
        # Use 'part_processed_image' if available, otherwise fallback to path
        if 'part_processed_image' in self.results and self.results['part_processed_image'] is not None:
            pixmap = convert_np_to_qpixmap(self.results['part_processed_image'], target_size)
            self.part_image_display.setPixmap(pixmap)
        elif 'part_boundary_image_path' in self.results and os.path.exists(self.results['part_boundary_image_path']):
            pixmap = QPixmap(self.results['part_boundary_image_path']).scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.part_image_display.setPixmap(pixmap)
        else:
            self.part_image_display.setText("N/A")
            logger.warning("Part boundary image not found in results or path.")


        if 'reference_boundary_image_path' in self.results and os.path.exists(self.results['reference_boundary_image_path']):
            pixmap = QPixmap(self.results['reference_boundary_image_path']).scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.reference_image_display.setPixmap(pixmap)
        else:
            self.reference_image_display.setText("N/A")
            logger.warning("Reference boundary image not found in results or path.")

        if 'superimposed_image_path' in self.results and os.path.exists(self.results['superimposed_image_path']):
            pixmap = QPixmap(self.results['superimposed_image_path']).scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.superimposed_image_display.setPixmap(pixmap)
        else:
            self.superimposed_image_display.setText("N/A")
            logger.warning("Superimposed image not found in results or path.")

        # Display metrics
        self.part_area_label.setText(f"{self.results.get('part_area_mm2', 0.0):.2f} mm²")
        self.reference_area_label.setText(f"{self.results.get('reference_area_mm2', 0.0):.2f} mm²")
        self.area_deviation_label.setText(f"{self.results.get('area_deviation_percent', 0.0):.2f} %")
        self.max_deviation_label.setText(f"{self.results.get('max_deviation_mm', 0.0):.2f} mm")

    def generate_report(self):
        try:
            # Prompt for project and operator name
            project_name, ok_project = QInputDialog.getText(self, "Report Details", "Enter Project Name (Optional):")
            if not ok_project: return # User cancelled

            operator_name, ok_operator = QInputDialog.getText(self, "Report Details", "Enter Operator Name (Optional):")
            if not ok_operator: return # User cancelled

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"Part_Analysis_Report_{timestamp}.pdf"

            report_generator = ReportGenerator(self.results, self.config)
            report_generator.generate_pdf_report(
                filename=report_filename,
                project_name=project_name,
                operator_name=operator_name
            )
            QMessageBox.information(self, "Report Generated", f"PDF report saved as:\n{report_filename}")
        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to generate report: {e}")


# --- Main Application Window ---
class PartAnalyzerApp(QWidget):
    """
    The main application window for the Part Analyzer Tool.
    Manages UI elements, user inputs, and triggers analysis.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Part Deviation Analyzer")
        self.setFixedSize(700, 750) # Fixed size for consistency
        self.analysis_worker = None
        self.comparison_window = None
        self.original_part_image = None # Store original part image for drawing scale
        self.current_part_contour = None # Store current part contour for set_scale

        self.init_ui()
        self.config = AnalysisConfig(image_path="", reference_path="") # Initialize with dummy paths

    def init_ui(self):
        main_layout = QVBoxLayout()
        # main_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align content to the top

        # --- File Selection Group ---
        file_selection_group = QGroupBox("File Selection")
        file_layout = QFormLayout()

        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("Select Part Image (JPG, BMP)")
        self.image_path_input.setReadOnly(True)
        self.select_image_button = QPushButton("Browse...")
        self.select_image_button.clicked.connect(self.select_image_file)
        image_h_layout = QHBoxLayout()
        image_h_layout.addWidget(self.image_path_input)
        image_h_layout.addWidget(self.select_image_button)
        file_layout.addRow("Part Image:", image_h_layout)

        self.reference_path_input = QLineEdit()
        self.reference_path_input.setPlaceholderText("Select Reference Image (JPG, BMP) or STL file")
        self.reference_path_input.setReadOnly(True)
        self.select_reference_button = QPushButton("Browse...")
        self.select_reference_button.clicked.connect(self.select_reference_file)
        reference_h_layout = QHBoxLayout()
        reference_h_layout.addWidget(self.reference_path_input)
        reference_h_layout.addWidget(self.select_reference_button)
        file_layout.addRow("Reference File:", reference_h_layout)

        # Reference Type selection
        self.reference_type_group = QHBoxLayout()
        self.radio_image_ref = QRadioButton("Image")
        self.radio_image_ref.setChecked(True)
        self.radio_image_ref.toggled.connect(self.update_reference_options)
        self.radio_stl_ref = QRadioButton("STL")
        self.radio_stl_ref.toggled.connect(self.update_reference_options)
        self.reference_type_group.addWidget(QLabel("Reference Type:"))
        self.reference_type_group.addWidget(self.radio_image_ref)
        self.reference_type_group.addWidget(self.radio_stl_ref)
        self.reference_type_group.addStretch(1) # Push radios to left
        file_layout.addRow(self.reference_type_group)

        # STL Projection Axis (initially hidden)
        self.projection_group = QHBoxLayout()
        self.radio_proj_x = QRadioButton("X-Axis")
        self.radio_proj_y = QRadioButton("Y-Axis")
        self.radio_proj_z = QRadioButton("Z-Axis")
        self.radio_proj_z.setChecked(True) # Default to Z-axis projection (top view)
        self.projection_group.addWidget(QLabel("STL Projection Axis:"))
        self.projection_group.addWidget(self.radio_proj_x)
        self.projection_group.addWidget(self.radio_proj_y)
        self.projection_group.addWidget(self.radio_proj_z)
        self.projection_group.addStretch(1)
        file_layout.addRow(self.projection_group)
        self.set_layout_visibility(self.projection_group, False) # Hide initially

        file_selection_group.setLayout(file_layout)
        main_layout.addWidget(file_selection_group)

        # --- Image Processing Parameters Group ---
        processing_group = QGroupBox("Image Processing Parameters")
        processing_layout = QFormLayout()

        # Processing Method selection
        self.processing_method_group = QHBoxLayout()
        self.radio_canny = QRadioButton("Canny Edge Detection")
        self.radio_canny.setChecked(True)
        self.radio_canny.toggled.connect(self.update_processing_inputs_visibility)
        self.radio_threshold = QRadioButton("Binary Thresholding")
        self.radio_threshold.toggled.connect(self.update_processing_inputs_visibility)
        self.processing_method_group.addWidget(self.radio_canny)
        self.processing_method_group.addWidget(self.radio_threshold)
        processing_layout.addRow("Method:", self.processing_method_group)

        # Canny parameters (initially visible)
        self.canny_threshold1_input = QLineEdit("50")
        self.canny_threshold1_input.setValidator(QIntValidator(0, 255))
        self.canny_threshold2_input = QLineEdit("150")
        self.canny_threshold2_input.setValidator(QIntValidator(0, 255))
        processing_layout.addRow("Canny Threshold 1:", self.canny_threshold1_input)
        processing_layout.addRow("Canny Threshold 2:", self.canny_threshold2_input)

        # Threshold parameter (initially hidden)
        self.threshold_value_input = QLineEdit("127")
        self.threshold_value_input.setValidator(QIntValidator(0, 255))
        processing_layout.addRow("Threshold Value:", self.threshold_value_input)
        # Hide threshold input initially
        for i in range(processing_layout.count()):
            item = processing_layout.itemAt(i)
            if item and item.widget() == self.threshold_value_input:
                processing_layout.labelForField(self.threshold_value_input).setVisible(False)
                self.threshold_value_input.setVisible(False)
                break

        # Blur Kernel
        self.blur_kernel_input = QLineEdit("5")
        self.blur_kernel_input.setValidator(QIntValidator(1, 999)) # Allow larger odd numbers
        processing_layout.addRow("Blur Kernel Size (Odd):", self.blur_kernel_input)

        processing_group.setLayout(processing_layout)
        main_layout.addWidget(processing_group)

        # --- Calibration Group ---
        calibration_group = QGroupBox("Scale Calibration (Pixels/mm)")
        calibration_layout = QFormLayout()

        # Part Calibration
        self.part_pixels_per_mm_label = QLabel("Part Pixels/mm: N/A")
        self.part_set_scale_button = QPushButton("Set Scale (Part Image)")
        self.part_set_scale_button.clicked.connect(self.set_part_scale)
        part_scale_layout = QHBoxLayout()
        part_scale_layout.addWidget(self.part_pixels_per_mm_label)
        part_scale_layout.addWidget(self.part_set_scale_button)
        calibration_layout.addRow(part_scale_layout)

        # Reference Calibration (conditionally visible)
        self.reference_calibration_group = QVBoxLayout() # Use QVBoxLayout to manage visibility of multiple widgets
        self.reference_pixels_per_mm_label = QLabel("Reference Pixels/mm: N/A")
        self.reference_set_scale_button = QPushButton("Set Scale (Reference Image)")
        self.reference_set_scale_button.clicked.connect(self.set_reference_scale)
        reference_scale_layout = QHBoxLayout()
        reference_scale_layout.addWidget(self.reference_pixels_per_mm_label)
        reference_scale_layout.addWidget(self.reference_set_scale_button)
        self.reference_calibration_group.addLayout(reference_scale_layout)
        calibration_layout.addRow(self.reference_calibration_group)
        self.reference_calibration_group.setContentsMargins(0, 0, 0, 0) # Remove extra margins
        self.reference_calibration_group.setSpacing(0) # Remove extra spacing
        self.set_layout_visibility(self.reference_calibration_group, True) # Initially visible, will be hidden if STL

        calibration_group.setLayout(calibration_layout)
        main_layout.addWidget(calibration_group)

        # --- Progress Bar and Buttons ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Progress: %p%")
        main_layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()
        self.perform_analysis_button = QPushButton("Perform Analysis")
        self.perform_analysis_button.clicked.connect(self.perform_analysis)
        self.perform_analysis_button.setEnabled(False) # Disable until files are selected

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_ui)
        self.reset_button.setEnabled(True) # Always enabled to reset

        button_layout.addWidget(self.perform_analysis_button)
        button_layout.addWidget(self.reset_button)
        main_layout.addLayout(button_layout)

        main_layout.addStretch(1) # Pushes everything to the top

        self.setLayout(main_layout)

        # Initial UI state setup
        self.update_reference_options() # Call once to set initial state of reference options
        self.update_processing_inputs_visibility() # Call once to set initial visibility of processing inputs

    def set_layout_visibility(self, layout, visible):
        """Helper to set visibility for all widgets in a layout."""
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().setVisible(visible)
            elif item.layout():
                self.set_layout_visibility(item.layout(), visible) # Recurse for nested layouts

    def update_reference_options(self):
        is_stl = self.radio_stl_ref.isChecked()
        self.set_layout_visibility(self.projection_group, is_stl)
        self.set_layout_visibility(self.reference_calibration_group, not is_stl) # Hide reference calibration for STL

        # Update config reference type
        self.config.reference_type = 'stl' if is_stl else 'image'
        self.check_enable_analysis_button()

    def update_processing_inputs_visibility(self):
        is_canny = self.radio_canny.isChecked()

        # Toggle visibility for Canny inputs
        self.canny_threshold1_input.setVisible(is_canny)
        self.canny_threshold2_input.setVisible(is_canny)
        # Find and toggle labels
        for i in range(self.findChild(QGroupBox, "Image Processing Parameters").layout().count()):
            item = self.findChild(QGroupBox, "Image Processing Parameters").layout().itemAt(i)
            if item and item.widget() == self.canny_threshold1_input:
                self.findChild(QGroupBox, "Image Processing Parameters").layout().labelForField(self.canny_threshold1_input).setVisible(is_canny)
            if item and item.widget() == self.canny_threshold2_input:
                self.findChild(QGroupBox, "Image Processing Parameters").layout().labelForField(self.canny_threshold2_input).setVisible(is_canny)

        # Toggle visibility for Threshold input
        self.threshold_value_input.setVisible(not is_canny)
        for i in range(self.findChild(QGroupBox, "Image Processing Parameters").layout().count()):
            item = self.findChild(QGroupBox, "Image Processing Parameters").layout().itemAt(i)
            if item and item.widget() == self.threshold_value_input:
                self.findChild(QGroupBox, "Image Processing Parameters").layout().labelForField(self.threshold_value_input).setVisible(not is_canny)
        
        self.config.processing_method = 'Canny' if is_canny else 'Threshold'

    def select_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Part Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.image_path_input.setText(file_path)
            self.config.image_path = file_path
            self.original_part_image = cv2.imread(file_path) # Load original image for set_scale
            self.part_pixels_per_mm_label.setText("Part Pixels/mm: N/A (Not Set)") # Reset calibration on new image
            self.config.part_pixels_per_mm_value = None # Clear calibration value
            self.check_enable_analysis_button()

    def select_reference_file(self):
        if self.radio_image_ref.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
            if file_path:
                self.reference_path_input.setText(file_path)
                self.config.reference_path = file_path
                self.reference_pixels_per_mm_label.setText("Reference Pixels/mm: N/A (Not Set)") # Reset calibration
                self.config.reference_pixels_per_mm_value = None # Clear calibration value
                self.check_enable_analysis_button()
        elif self.radio_stl_ref.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference STL", "", "STL Files (*.stl)")
            if file_path:
                self.reference_path_input.setText(file_path)
                self.config.reference_path = file_path
                self.reference_pixels_per_mm_label.setText("Reference Pixels/mm: N/A (STL)") # STL doesn't use image calibration
                self.config.reference_pixels_per_mm_value = None # Clear calibration value for STL
                self.check_enable_analysis_button()

    def check_enable_analysis_button(self):
        # Enable analysis only if both paths are set
        is_ready = bool(self.image_path_input.text()) and bool(self.reference_path_input.text())
        self.perform_analysis_button.setEnabled(is_ready)
        self.part_set_scale_button.setEnabled(bool(self.image_path_input.text()))
        self.reference_set_scale_button.setEnabled(bool(self.reference_path_input.text()) and self.radio_image_ref.isChecked()) # Only for image ref

    def set_part_scale(self):
        self._set_scale_dialog(is_part_image=True)

    def set_reference_scale(self):
        if self.radio_image_ref.isChecked():
            self._set_scale_dialog(is_part_image=False)
        else:
            QMessageBox.information(self, "Scale Not Applicable", "Calibration is not applicable for STL reference files as they are intrinsically dimensioned.")


    def _set_scale_dialog(self, is_part_image: bool):
        image_path = self.image_path_input.text() if is_part_image else self.reference_path_input.text()
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", "Please select an image file first.")
            return

        try:
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError("Could not load image for calibration.")

            # Temporarily apply current processing settings to get a contour
            temp_config = AnalysisConfig(
                image_path=image_path,
                reference_path="", # Dummy
                processing_method=self.config.processing_method,
                canny_threshold1=int(self.canny_threshold1_input.text()),
                canny_threshold2=int(self.canny_threshold2_input.text()),
                threshold_value=int(self.threshold_value_input.text()),
                blur_kernel=int(self.blur_kernel_input.text())
            )
            from .image_processing import ImageProcessor # Import locally to avoid circular
            temp_processor = ImageProcessor(temp_config)
            
            # Use load_and_process_image to get the processed image and contour
            _, _, processed_binary_image, temp_contour = temp_processor.load_and_process_image(image_path)

            if temp_contour.size == 0:
                QMessageBox.warning(self, "No Contour", "Could not detect a clear contour in the image with current settings. Adjust processing parameters or use a different image.")
                return

            # Draw the contour on a copy of the original image for display in dialog
            display_image = original_image.copy()
            if len(display_image.shape) == 2: # Ensure it's BGR for color drawing
                display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
            
            # Draw the detected contour in a prominent color (e.g., yellow)
            cv2.drawContours(display_image, [temp_contour], -1, (0, 255, 255), 2) # Yellow contour

            dialog = ScaleCalibrationDialog(display_image, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                p1 = dialog.start_point
                p2 = dialog.end_point
                known_distance_mm, ok = QInputDialog.getDouble(self, "Known Distance", "Enter known distance in mm:", 10.0, 0.01, 1000.0, 2)

                if ok and p1 and p2:
                    pixel_distance = np.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
                    if pixel_distance > 0:
                        pixels_per_mm = pixel_distance / known_distance_mm
                        if is_part_image:
                            self.config.part_pixels_per_mm_value = pixels_per_mm
                            self.part_pixels_per_mm_label.setText(f"Part Pixels/mm: {pixels_per_mm:.2f}")
                        else:
                            self.config.reference_pixels_per_mm_value = pixels_per_mm
                            self.reference_pixels_per_mm_label.setText(f"Reference Pixels/mm: {pixels_per_mm:.2f}")
                        QMessageBox.information(self, "Scale Set", f"Pixels/mm set to {pixels_per_mm:.2f}")
                    else:
                        QMessageBox.warning(self, "Error", "Clicked points are too close or identical.")
                else:
                    QMessageBox.warning(self, "Cancelled", "Calibration cancelled or invalid distance entered.")
            else:
                QMessageBox.information(self, "Cancelled", "Calibration cancelled by user.")

        except Exception as e:
            logger.error(f"Error in scale calibration: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to set scale: {e}")


    def get_config_from_ui(self):
        """Collects current UI settings into an AnalysisConfig object."""
        try:
            blur_k = int(self.blur_kernel_input.text())
            if blur_k % 2 == 0:
                raise ValueError("Blur kernel must be an odd number.")

            config = AnalysisConfig(
                image_path=self.image_path_input.text(),
                reference_path=self.reference_path_input.text(),
                reference_type='stl' if self.radio_stl_ref.isChecked() else 'image',
                projection_axis='X' if self.radio_proj_x.isChecked() else ('Y' if self.radio_proj_y.isChecked() else 'Z'),
                threshold_value=int(self.threshold_value_input.text()) if self.radio_threshold.isChecked() else 0, # Only relevant if thresholding
                canny_threshold1=int(self.canny_threshold1_input.text()) if self.radio_canny.isChecked() else 0, # Only relevant if Canny
                canny_threshold2=int(self.canny_threshold2_input.text()) if self.radio_canny.isChecked() else 0, # Only relevant if Canny
                blur_kernel=blur_k,
                processing_method='Canny' if self.radio_canny.isChecked() else 'Threshold',
                part_pixels_per_mm_value=self.config.part_pixels_per_mm_value,
                reference_pixels_per_mm_value=self.config.reference_pixels_per_mm_value
            )
            return config
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input for parameters: {e}")
            return None
        except Exception as e:
            logger.error(f"Error collecting config from UI: {e}", exc_info=True)
            QMessageBox.critical(self, "Configuration Error", f"Failed to get configuration: {e}")
            return None


    def perform_analysis(self):
        config = self.get_config_from_ui()
        if config is None:
            return # get_config_from_ui already showed an error message

        # Store the current config in the app for passing to comparison window
        self.config = config

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Analysis in progress: %p%")
        self.perform_analysis_button.setEnabled(False)
        self.reset_button.setEnabled(False)

        # Start the analysis worker thread
        self.analysis_worker = AnalysisWorker(config)
        self.analysis_worker.analysis_completed.connect(self.analysis_finished)
        self.analysis_worker.analysis_error.connect(self.analysis_failed)
        self.analysis_worker.progress_updated_with_text.connect(self.update_progress)
        self.analysis_worker.start()

    def analysis_finished(self, results):
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Analysis Complete!")
        self.perform_analysis_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        QMessageBox.information(self, "Analysis Complete", "Deviation analysis completed successfully!")
        
        # Display results in a new window
        self.comparison_window = BoundaryComparisonWindow(results, self.config) # Pass config here
        self.comparison_window.show()

    def analysis_failed(self, error_message):
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Analysis Failed")
        self.perform_analysis_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", error_message)
        logger.error(f"Analysis failed: {error_message}")

    def update_progress(self, value, text):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{text}: %p%")

    def reset_ui(self):
        # Clear file paths and reset labels
        self.image_path_input.clear()
        self.reference_path_input.clear()
        self.part_pixels_per_mm_label.setText("Part Pixels/mm: N/A (Not Set)")
        self.reference_pixels_per_mm_label.setText("Reference Pixels/mm: N/A (Not Set)") # Reset calibration label
        self.config.part_pixels_per_mm_value = None # Clear config value
        self.config.reference_pixels_per_mm_value = None # Clear config value

        # Reset radio buttons to default
        self.radio_image_ref.setChecked(True)
        self.radio_proj_z.setChecked(True) # Default STL projection to Z
        self.reference_pixels_per_mm_label.setText("Reference Pixels/mm: N/A")
        self.reference_set_scale_button.setEnabled(False)

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
            self.analysis_worker.wait() # Wait for it to finish gracefully
            logger.info("Analysis worker stopped during reset.")

        if self.comparison_window:
            self.comparison_window.close()
            self.comparison_window = None

        self.reference_calibration_group.setVisible(False) # Hide reference calibration on reset
        self.progress_bar.setFormat("Progress: %p%") # Reset format
        self.check_enable_analysis_button() # Re-check state for analysis button

class ScaleCalibrationDialog(QDialog):
    """
    A dialog for users to click two points on an image to define a known distance.
    """
    def __init__(self, image_np, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate Scale: Click Two Points")
        self.setFixedSize(800, 600) # Fixed size for the dialog
        
        self.image_np = image_np
        self.start_point = None
        self.end_point = None
        self.points = []

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setScaledContents(True) # Scale image to fit label size
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label.setMouseTracking(True) # Enable mouse tracking for live line drawing
        self.label.paintEvent = self.paintEvent # Override paintEvent for drawing

        self.original_pixmap = convert_np_to_qpixmap(self.image_np)
        self.label.setPixmap(self.original_pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        self.label.mousePressEvent = self.mouse_press
        self.label.mouseMoveEvent = self.mouse_move # Connect mouse move event

        # Add instructions
        instructions = QLabel("Click two points on the image to define a known distance. Press 'Done' when finished.")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.done_button = QPushButton("Done")
        self.done_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(instructions)
        layout.addWidget(self.label)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.done_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.update_label_pixmap()

    def resizeEvent(self, event):
        # Override resizeEvent to scale the pixmap when the window resizes
        super().resizeEvent(event)
        self.update_label_pixmap()

    def update_label_pixmap(self):
        if not self.original_pixmap.isNull():
            self.label.setPixmap(self.original_pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


    def paintEvent(self, event):
        """Custom paint event to draw points and lines on the QLabel."""
        painter = QPainter(self.label) # Paint on the QLabel itself
        # First, draw the pixmap background
        pixmap_to_draw = self.label.pixmap()
        if pixmap_to_draw:
            painter.drawPixmap(self.label.rect(), pixmap_to_draw)

        # Set pen for drawing points and lines
        pen = QPen(QColor(255, 0, 0)) # Red color
        pen.setWidth(3)
        painter.setPen(pen)

        # Draw clicked points
        for p in self.points:
            painter.drawEllipse(p, 5, 5) # Draw a circle at each point

        # Draw line segment between start and end points
        if self.start_point and self.end_point:
            painter.drawLine(QLineF(self.start_point, self.end_point))
            
        painter.end() # Crucial to end the painter

    def mouse_press(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.start_point:
                self.start_point = event.pos()
                self.points.append(self.start_point)
            else:
                self.end_point = event.pos()
                self.points.append(self.end_point)
                self.label.unsetMouseTracking() # Stop tracking after two points are set
            self.label.update() # Trigger repaint

    def mouse_move(self, event):
        if self.start_point and not self.end_point:
            self.end_point = event.pos() # Temporarily update end_point for live drawing
            self.label.update() # Trigger repaint
        elif not self.start_point:
            # If no start point, show cursor position for potential first click
            pass # No need to draw a line yet