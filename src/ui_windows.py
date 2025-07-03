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
        self.setFixedSize(900, 650)
        self.results = results
        self.config = config # Store config

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        image_display_layout = QHBoxLayout()

        self.part_image_label = QLabel("Part Boundary")
        self.part_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.part_image_label.setFixedSize(250, 250)
        self.part_image_label.setStyleSheet("border: 1px solid black;")

        self.reference_image_label = QLabel("Reference Boundary")
        self.reference_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reference_image_label.setFixedSize(250, 250)
        self.reference_image_label.setStyleSheet("border: 1px solid black;")

        self.superimposed_image_label = QLabel("Superimposed View")
        self.superimposed_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.superimposed_image_label.setFixedSize(250, 250)
        self.superimposed_image_label.setStyleSheet("border: 1px solid black;")

        image_display_layout.addWidget(self.part_image_label)
        image_display_layout.addWidget(self.reference_image_label)
        image_display_layout.addWidget(self.superimposed_image_label)

        metrics_layout = QVBoxLayout()
        self.area_deviation_label = QLabel("Area Deviation: N/A")
        self.max_deviation_label = QLabel("Max Contour Deviation: N/A")
        
        metrics_layout.addWidget(self.area_deviation_label)
        metrics_layout.addWidget(self.max_deviation_label)
        metrics_layout.addStretch()

        # Add a "Generate Report" button
        self.generate_report_button = QPushButton("Generate PDF Report")
        self.generate_report_button.clicked.connect(self.generate_report)
        metrics_layout.addWidget(self.generate_report_button)

        main_layout.addLayout(image_display_layout)
        main_layout.addLayout(metrics_layout)
        self.setLayout(main_layout)

        self.load_images_and_metrics()

    def load_images_and_metrics(self):
        # Display part image with its contour
        if 'part_contour_original_for_display' in self.results:
            pixmap = convert_np_to_qpixmap(self.results['part_contour_original_for_display'], self.part_image_label.size())
            self.part_image_label.setPixmap(pixmap)
        
        # Display reference image with its contour/projection
        if 'reference_contour_original_for_display' in self.results:
            pixmap = convert_np_to_qpixmap(self.results['reference_contour_original_for_display'], self.reference_image_label.size())
            self.reference_image_label.setPixmap(pixmap)

        # Display superimposed image
        superimposed_path = self.results.get('superimposed_image_path')
        if superimposed_path and os.path.exists(superimposed_path):
            pixmap = QPixmap(superimposed_path)
            self.superimposed_image_label.setPixmap(pixmap.scaled(self.superimposed_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.superimposed_image_label.setText("Superimposed Image Not Found")

        # Display metrics
        self.area_deviation_label.setText(f"Area Deviation: {self.results.get('area_deviation_percent', 0.0):.2f} %")
        self.max_deviation_label.setText(f"Max Contour Deviation: {self.results.get('max_deviation_mm', 0.0):.2f} mm")

    def generate_report(self):
        try:
            # Prompt for project name and operator name
            project_name, ok_project = QInputDialog.getText(self, "Report Details", "Enter Project Name (Optional):")
            if not ok_project: return # User cancelled

            operator_name, ok_operator = QInputDialog.getText(self, "Report Details", "Enter Operator Name (Optional):")
            if not ok_operator: return # User cancelled

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"Part_Analysis_Report_{timestamp}.pdf"
            
            # Use QFileDialog to let the user choose where to save the report
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Report", default_filename, "PDF Files (*.pdf)")
            
            if filepath:
                report_gen = ReportGenerator(self.results, self.config)
                report_gen.generate_pdf_report(filename=filepath, project_name=project_name, operator_name=operator_name)
                QMessageBox.information(self, "Report Generated", f"Report saved successfully to:\n{filepath}")
            else:
                QMessageBox.warning(self, "Save Cancelled", "Report generation cancelled.")

        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            QMessageBox.critical(self, "Report Error", f"Failed to generate report: {e}")

class PartAnalyzerApp(QWidget):
    """
    Main application window for the Part Analyzer Tool.
    Manages UI elements for file selection, parameter input, analysis, and result display.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Part Analyzer Tool - Alpha v1")
        self.setFixedSize(700, 750) # Adjusted size to fit new elements

        self.part_image_path = None
        self.reference_path = None
        self.reference_type = 'image' # Default reference type
        self.analysis_worker = None
        self.comparison_window = None # To hold the reference to the comparison window

        # Calibration values, can be set by user or derived
        self.part_pixels_per_mm = None
        self.reference_pixels_per_mm = None

        self.init_ui()
        self.reset_application_state() # Initialize UI elements and state

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # --- File Selection Group ---
        file_selection_group = QGroupBox("File Selection")
        file_layout = QGridLayout()

        self.part_image_label = QLabel("Part Image (JPG/BMP):")
        self.part_image_path_display = QLineEdit()
        self.part_image_path_display.setReadOnly(True)
        self.load_part_image_button = QPushButton("Load Part Image")
        self.load_part_image_button.clicked.connect(self.load_part_image)
        self.part_set_scale_button = QPushButton("Set Scale (Part)")
        self.part_set_scale_button.clicked.connect(self.set_part_scale)
        self.part_set_scale_button.setEnabled(False) # Initially disabled

        self.reference_file_label = QLabel("Reference (Image/STL):")
        self.reference_file_path_display = QLineEdit()
        self.reference_file_path_display.setReadOnly(True)
        self.load_reference_file_button = QPushButton("Load Reference File")
        self.load_reference_file_button.clicked.connect(self.load_reference_file)

        # Reference Type Radio Buttons
        self.reference_type_group = QGroupBox("Reference Type")
        type_layout = QHBoxLayout()
        self.radio_image = QRadioButton("Image")
        self.radio_image.setChecked(True)
        self.radio_image.toggled.connect(lambda: self.set_reference_type('image'))
        self.radio_stl = QRadioButton("STL (3D Model)")
        self.radio_stl.toggled.connect(lambda: self.set_reference_type('stl'))
        type_layout.addWidget(self.radio_image)
        type_layout.addWidget(self.radio_stl)
        self.reference_type_group.setLayout(type_layout)

        # STL Projection Axis selection (initially hidden/disabled)
        self.projection_group = QGroupBox("STL Projection Axis")
        proj_layout = QHBoxLayout()
        self.radio_proj_x = QRadioButton("X-Axis")
        self.radio_proj_y = QRadioButton("Y-Axis")
        self.radio_proj_z = QRadioButton("Z-Axis")
        self.radio_proj_z.setChecked(True) # Default for STL
        proj_layout.addWidget(self.radio_proj_x)
        proj_layout.addWidget(self.radio_proj_y)
        proj_layout.addWidget(self.radio_proj_z)
        self.projection_group.setLayout(proj_layout)
        self.projection_group.setEnabled(False) # Disable by default

        # Reference Calibration Group
        self.reference_calibration_group = QGroupBox("Reference Calibration")
        ref_cal_layout = QHBoxLayout()
        self.reference_pixels_per_mm_label = QLabel("Reference Pixels/mm: N/A")
        self.reference_set_scale_button = QPushButton("Set Scale (Reference)")
        self.reference_set_scale_button.clicked.connect(self.set_reference_scale)
        self.reference_set_scale_button.setEnabled(False) # Initially disabled
        ref_cal_layout.addWidget(self.reference_pixels_per_mm_label)
        ref_cal_layout.addWidget(self.reference_set_scale_button)
        self.reference_calibration_group.setLayout(ref_cal_layout)
        self.reference_calibration_group.setVisible(False) # Hide by default

        file_layout.addWidget(self.part_image_label, 0, 0)
        file_layout.addWidget(self.part_image_path_display, 0, 1)
        file_layout.addWidget(self.load_part_image_button, 0, 2)
        file_layout.addWidget(self.part_set_scale_button, 0, 3)

        file_layout.addWidget(self.reference_file_label, 1, 0)
        file_layout.addWidget(self.reference_file_path_display, 1, 1)
        file_layout.addWidget(self.load_reference_file_button, 1, 2)
        file_layout.addWidget(self.reference_type_group, 2, 0, 1, 2) # Span two columns
        file_layout.addWidget(self.projection_group, 2, 2, 1, 2) # Span two columns
        file_layout.addWidget(self.reference_calibration_group, 3, 0, 1, 4) # Span all columns

        file_selection_group.setLayout(file_layout)
        main_layout.addWidget(file_selection_group)

        # --- Image Processing Parameters Group ---
        params_group = QGroupBox("Image Processing Parameters")
        params_layout = QFormLayout()

        # Processing Method Radio Buttons
        self.processing_method_group = QGroupBox("Method")
        method_layout = QHBoxLayout()
        self.radio_canny = QRadioButton("Canny Edge Detection")
        self.radio_canny.setChecked(True)
        self.radio_canny.toggled.connect(self.update_processing_inputs_visibility)
        self.radio_threshold = QRadioButton("Binary Thresholding")
        self.radio_threshold.toggled.connect(self.update_processing_inputs_visibility)
        method_layout.addWidget(self.radio_canny)
        method_layout.addWidget(self.radio_threshold)
        self.processing_method_group.setLayout(method_layout)
        params_layout.addRow(self.processing_method_group)

        # Canny inputs
        self.canny_threshold1_input = QLineEdit("50")
        self.canny_threshold1_input.setValidator(QIntValidator(0, 255))
        params_layout.addRow("Canny Threshold 1 (0-255):", self.canny_threshold1_input)

        self.canny_threshold2_input = QLineEdit("150")
        self.canny_threshold2_input.setValidator(QIntValidator(0, 255))
        params_layout.addRow("Canny Threshold 2 (0-255):", self.canny_threshold2_input)

        # Threshold input
        self.threshold_value_input = QLineEdit("127")
        self.threshold_value_input.setValidator(QIntValidator(0, 255))
        params_layout.addRow("Threshold Value (0-255):", self.threshold_value_input)
        self.threshold_value_input.hide() # Hide by default, shown if Threshold method selected

        # Blur kernel size
        self.blur_kernel_input = QLineEdit("5")
        self.blur_kernel_input.setValidator(QIntValidator(1, 99)) # Allow odd numbers
        params_layout.addRow("Gaussian Blur Kernel (Odd):", self.blur_kernel_input)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # --- Analysis and Reset Buttons ---
        button_layout = QHBoxLayout()
        self.perform_analysis_button = QPushButton("Perform Analysis")
        self.perform_analysis_button.clicked.connect(self.perform_analysis)
        self.perform_analysis_button.setEnabled(False) # Initially disabled

        self.reset_button = QPushButton("Reset Application")
        self.reset_button.clicked.connect(self.reset_application_state)

        button_layout.addWidget(self.perform_analysis_button)
        button_layout.addWidget(self.reset_button)
        main_layout.addLayout(button_layout)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setFormat("Progress: %p%")
        main_layout.addWidget(self.progress_bar)

        main_layout.addStretch() # Pushes everything to the top
        self.setLayout(main_layout)

        self.update_processing_inputs_visibility() # Call once to set initial state

    def update_processing_inputs_visibility(self):
        """Toggles visibility of Canny/Threshold inputs based on radio button selection."""
        is_canny = self.radio_canny.isChecked()
        self.canny_threshold1_input.setVisible(is_canny)
        self.canny_threshold1_input.setParent(None if not is_canny else self.findChild(QWidget, "qt_spinbox_Canny Threshold 1 (0-255):")) # Hack to hide from layout
        self.canny_threshold2_input.setVisible(is_canny)
        self.canny_threshold2_input.setParent(None if not is_canny else self.findChild(QWidget, "qt_spinbox_Canny Threshold 2 (0-255):"))

        self.threshold_value_input.setVisible(not is_canny)
        self.threshold_value_input.setParent(None if is_canny else self.findChild(QWidget, "qt_spinbox_Threshold Value (0-255):"))

        # Re-add widgets to layout if they were removed, needed for QFormLayout
        form_layout = self.findChild(QFormLayout)
        if form_layout:
            # Clear existing rows to rebuild dynamically (this is a bit heavy but works for simple cases)
            # A more robust solution might use QStackedLayout or just hide/show items within existing rows if QFormLayout allowed it easily.
            # For this context, we will directly control visibility which is handled by setVisible and setParent.
            # The .setParent(None) trick removes it from layout management.
            # We don't need to explicitly re-add to QFormLayout rows when just changing visibility.
            pass


    def load_part_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Image files (*.jpg *.jpeg *.bmp *.png)") # Added PNG
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            if filenames:
                self.part_image_path = filenames[0]
                self.part_image_path_display.setText(self.part_image_path)
                self.part_set_scale_button.setEnabled(True) # Enable scale button once image is loaded
                logger.info(f"Part image loaded: {self.part_image_path}")
                self.check_enable_analysis_button()

    def load_reference_file(self):
        file_dialog = QFileDialog(self)
        # Allow both image and STL files for reference
        file_dialog.setNameFilter("Reference files (*.jpg *.jpeg *.bmp *.png *.stl);;Image files (*.jpg *.jpeg *.bmp *.png);;STL files (*.stl)")
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            if filenames:
                self.reference_path = filenames[0]
                self.reference_file_path_display.setText(self.reference_path)
                
                # Determine file type based on extension
                _, ext = os.path.splitext(self.reference_path)
                if ext.lower() in ('.stl'):
                    self.reference_type = 'stl'
                    self.radio_stl.setChecked(True) # Update radio button
                    self.projection_group.setEnabled(True) # Enable STL projection options
                    self.reference_calibration_group.setVisible(False) # Hide calibration for STL
                elif ext.lower() in ('.jpg', '.jpeg', '.bmp', '.png'):
                    self.reference_type = 'image'
                    self.radio_image.setChecked(True) # Update radio button
                    self.projection_group.setEnabled(False) # Disable STL projection options
                    self.reference_calibration_group.setVisible(True) # Show calibration for image reference
                    self.reference_set_scale_button.setEnabled(True) # Enable scale button for image reference
                else:
                    QMessageBox.warning(self, "Unsupported File Type", "Selected file type is not supported as a reference.")
                    self.reference_path = None
                    self.reference_file_path_display.clear()
                    self.reference_type = 'image' # Reset to default
                    self.radio_image.setChecked(True)
                    self.projection_group.setEnabled(False)
                    self.reference_calibration_group.setVisible(False)
                    self.reference_set_scale_button.setEnabled(False)
                    logger.warning(f"Unsupported reference file type selected: {ext}")
                    return

                logger.info(f"Reference file loaded: {self.reference_path} (Type: {self.reference_type})")
                self.check_enable_analysis_button()

    def set_reference_type(self, r_type):
        """Sets the reference type based on radio button selection."""
        self.reference_type = r_type
        self.projection_group.setEnabled(r_type == 'stl')
        self.reference_calibration_group.setVisible(r_type == 'image')
        # If switching from STL to Image, re-enable reference scale button if a file is loaded
        if r_type == 'image' and self.reference_path and os.path.splitext(self.reference_path)[1].lower() in ('.jpg', '.jpeg', '.bmp', '.png'):
            self.reference_set_scale_button.setEnabled(True)
        else:
            self.reference_set_scale_button.setEnabled(False)
        logger.info(f"Reference type set to: {self.reference_type}")


    def set_part_scale(self):
        if not self.part_image_path:
            QMessageBox.warning(self, "No Image", "Please load a part image first.")
            return

        scale_value, ok = QInputDialog.getDouble(self, "Set Part Image Scale",
                                              "Enter pixels per mm for Part Image:",
                                              self.part_pixels_per_mm if self.part_pixels_per_mm else 1.0, # Default value
                                              0.01, 1000.0, 2) # Min, Max, Decimals
        if ok:
            self.part_pixels_per_mm = scale_value
            QMessageBox.information(self, "Scale Set", f"Part image scale set to: {self.part_pixels_per_mm:.2f} pixels/mm")
            logger.info(f"Part image scale set to: {self.part_pixels_per_mm:.2f} pixels/mm (User Input)")
            self.check_enable_analysis_button()

    def set_reference_scale(self):
        if not self.reference_path or self.reference_type != 'image':
            QMessageBox.warning(self, "No Image", "Please load a reference image first, and ensure its type is 'Image'.")
            return

        scale_value, ok = QInputDialog.getDouble(self, "Set Reference Image Scale",
                                              "Enter pixels per mm for Reference Image:",
                                              self.reference_pixels_per_mm if self.reference_pixels_per_mm else 1.0, # Default value
                                              0.01, 1000.0, 2)
        if ok:
            self.reference_pixels_per_mm = scale_value
            self.reference_pixels_per_mm_label.setText(f"Reference Pixels/mm: {self.reference_pixels_per_mm:.2f}")
            QMessageBox.information(self, "Scale Set", f"Reference image scale set to: {self.reference_pixels_per_mm:.2f} pixels/mm")
            logger.info(f"Reference image scale set to: {self.reference_pixels_per_mm:.2f} pixels/mm (User Input)")
            self.check_enable_analysis_button()

    def check_enable_analysis_button(self):
        """Enables the 'Perform Analysis' button if both part and reference paths are set."""
        if self.part_image_path and self.reference_path:
            self.perform_analysis_button.setEnabled(True)
        else:
            self.perform_analysis_button.setEnabled(False)

    def perform_analysis(self):
        if not self.part_image_path or not self.reference_path:
            QMessageBox.warning(self, "Missing Files", "Please load both part image and reference file before analysis.")
            return

        # Basic validation for blur kernel size
        try:
            blur_k = int(self.blur_kernel_input.text())
            if blur_k % 2 == 0 or blur_k <= 0:
                QMessageBox.warning(self, "Invalid Input", "Blur Kernel Size must be a positive odd number.")
                return
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Blur Kernel Size must be an integer.")
            return

        # Create a dictionary for configuration to pass to AnalysisWorker
        config = AnalysisConfig(
            image_path=self.part_image_path,
            reference_path=self.reference_path,
            reference_type=self.reference_type,
            projection_axis='X' if self.radio_proj_x.isChecked() else ('Y' if self.radio_proj_y.isChecked() else 'Z'),
            threshold_value=int(self.threshold_value_input.text()),
            canny_threshold1=int(self.canny_threshold1_input.text()),
            canny_threshold2=int(self.canny_threshold2_input.text()),
            blur_kernel=blur_k,
            processing_method='Canny' if self.radio_canny.isChecked() else 'Threshold',
            part_pixels_per_mm_value=self.part_pixels_per_mm,
            reference_pixels_per_mm_value=self.reference_pixels_per_mm
        )
        logger.info(f"Analysis config created: {config}")
        
        self.perform_analysis_button.setEnabled(False) # Disable button during analysis
        self.reset_button.setEnabled(False) # Disable reset during analysis
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Analysis in Progress: %p%")
        logger.info("Analysis started.")

        # Initialize and start the worker thread
        self.analysis_worker = AnalysisWorker(config)
        self.analysis_worker.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_worker.analysis_error.connect(self.on_analysis_error)
        self.analysis_worker.progress_updated_with_text.connect(self.update_progress_bar)
        self.analysis_worker.start()

    def update_progress_bar(self, value, text):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{text} %p%")
        logger.info(f"{text}")

    def on_analysis_completed(self, results):
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Analysis Complete!")
        QMessageBox.information(self, "Analysis Complete", "Part deviation analysis finished successfully!")
        self.perform_analysis_button.setEnabled(True) # Re-enable button
        self.reset_button.setEnabled(True) # Re-enable reset

        # Open the comparison window with results
        if self.comparison_window and self.comparison_window.isVisible():
            self.comparison_window.close() # Close existing window if open
        self.comparison_window = BoundaryComparisonWindow(results, self.analysis_worker.config) # Pass config also
        self.comparison_window.show()

    def on_analysis_error(self, error_message):
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Analysis Failed")
        QMessageBox.critical(self, "Analysis Error", error_message)
        self.perform_analysis_button.setEnabled(True) # Re-enable button
        self.reset_button.setEnabled(True) # Re-enable reset

    def reset_application_state(self):
        logger.info("Resetting application state.")
        self.part_image_path = None
        self.reference_path = None
        self.reference_type = 'image'
        self.part_pixels_per_mm = None
        self.reference_pixels_per_mm = None

        self.part_image_path_display.clear()
        self.reference_file_path_display.clear()
        self.part_set_scale_button.setEnabled(False)
        self.radio_image.setChecked(True)
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
        logger.info("Application state reset complete.")