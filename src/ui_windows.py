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
    QScrollArea, QDialog, QFormLayout, QInputDialog,
    QMainWindow # Add QMainWindow here
)
from PyQt6.QtGui import QPixmap, QImage, QDoubleValidator, QIntValidator, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QPoint, QLineF, QPointF # Import QPointF

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
        super().__init__(parent)
        self.results = results
        self.config = config # Store config for report generation
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Boundary Comparison Results")
        self.setGeometry(100, 100, 1000, 800)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Labels to display images
        self.part_image_label = QLabel("Part Boundary")
        self.reference_image_label = QLabel("Reference Boundary")
        self.superimposed_image_label = QLabel("Superimposed (Aligned)")

        self.part_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reference_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.superimposed_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Group box for images
        image_group_box = QGroupBox("Visual Results")
        image_layout = QHBoxLayout()
        image_group_box.setLayout(image_layout)

        image_layout.addWidget(self.part_image_label)
        image_layout.addWidget(self.reference_image_label)
        image_layout.addWidget(self.superimposed_image_label)
        
        main_layout.addWidget(image_group_box)

        # Quantitative Metrics Display
        metrics_group_box = QGroupBox("Quantitative Metrics")
        metrics_layout = QFormLayout()
        metrics_group_box.setLayout(metrics_layout)

        self.area_deviation_label = QLabel("Area Deviation: N/A")
        self.max_deviation_label = QLabel("Maximum Contour Deviation: N/A")

        metrics_layout.addRow("Area Deviation:", self.area_deviation_label)
        metrics_layout.addRow("Max Contour Deviation:", self.max_deviation_label)

        main_layout.addWidget(metrics_group_box)

        # Add a "Generate Report" button
        self.generate_report_button = QPushButton("Generate PDF Report")
        self.generate_report_button.clicked.connect(self.generate_report)
        main_layout.addWidget(self.generate_report_button)

        self.display_results()

    def display_results(self):
        # Display images
        image_display_size = QSize(300, 300) # Standardize display size

        part_processed_image = self.results.get('part_processed_display_image')
        if part_processed_image is not None:
            self.part_image_label.setPixmap(convert_np_to_qpixmap(part_processed_image, image_display_size))
        else:
            self.part_image_label.setText("Part boundary image not found.")

        reference_processed_image = self.results.get('reference_processed_display_image')
        if reference_processed_image is not None:
            self.reference_image_label.setPixmap(convert_np_to_qpixmap(reference_processed_image, image_display_size))
        else:
            self.reference_image_label.setText("Reference boundary image not found.")

        superimposed_image = self.results.get('superimposed_image')
        if superimposed_image is not None:
            self.superimposed_image_label.setPixmap(convert_np_to_qpixmap(superimposed_image, image_display_size))
        else:
            self.superimposed_image_label.setText("Superimposed image not found.")

        # Display metrics
        area_dev = self.results.get('area_deviation_percent', 0.0)
        max_dev = self.results.get('max_deviation_mm', 0.0)

        self.area_deviation_label.setText(f"{area_dev:.2f}%")
        self.max_deviation_label.setText(f"{max_dev:.2f} mm")

    def generate_report(self):
        try:
            # Prompt user for project name and operator name
            project_name, ok_project = QInputDialog.getText(self, "Report Details", "Enter Project Name:", QLineEdit.EchoMode.Normal, "Part Analysis")
            if not ok_project:
                return # User cancelled

            operator_name, ok_operator = QInputDialog.getText(self, "Report Details", "Enter Operator Name:", QLineEdit.EchoMode.Normal, "User")
            if not ok_operator:
                return # User cancelled

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"Part_Analysis_Report_{timestamp}.pdf"
            
            # Prompt user to save PDF
            file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", default_filename, "PDF Files (*.pdf)")

            if file_path:
                report_generator = ReportGenerator(self.results, self.config)
                report_generator.generate_pdf_report(
                    filename=file_path,
                    project_name=project_name,
                    operator_name=operator_name
                )
                QMessageBox.information(self, "Report Generated", f"PDF report saved to:\n{file_path}")
            else:
                QMessageBox.warning(self, "Save Cancelled", "PDF report generation cancelled.")

        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to generate report: {e}")

# --- Main Application Window ---
class PartAnalyzerApp(QMainWindow): # Changed from QWidget to QMainWindow
    """
    The main application window for the Part Analyzer tool.
    """
    def __init__(self):
        super().__init__()
        self.analysis_worker = None
        self.comparison_window = None
        self.init_ui()
        self.reset_ui_state() # Call reset to set initial state

    def init_ui(self):
        self.setWindowTitle("Part Analyzer - Alpha v1")
        self.setGeometry(100, 100, 1200, 900) # Adjusted size for better layout

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # --- File Selection Group ---
        file_selection_group = QGroupBox("File Selection")
        file_selection_layout = QFormLayout()
        file_selection_group.setLayout(file_selection_layout)

        # Part Image
        self.part_path_input = QLineEdit()
        self.part_path_input.setPlaceholderText("Select part image (JPG, BMP)")
        self.part_path_input.setReadOnly(True)
        self.browse_part_button = QPushButton("Browse Part Image")
        self.browse_part_button.clicked.connect(self.browse_part_image)
        file_selection_layout.addRow("Part Image:", self.part_path_input)
        file_selection_layout.addWidget(self.browse_part_button)

        # Reference File
        self.reference_path_input = QLineEdit()
        self.reference_path_input.setPlaceholderText("Select reference image (JPG, BMP) or STL file")
        self.reference_path_input.setReadOnly(True)
        self.browse_reference_button = QPushButton("Browse Reference File")
        self.browse_reference_button.clicked.connect(self.browse_reference_file)
        file_selection_layout.addRow("Reference File:", self.reference_path_input)
        file_selection_layout.addWidget(self.browse_reference_button)

        main_layout.addWidget(file_selection_group)

        # --- Processing Parameters Group ---
        processing_group = QGroupBox("Processing Parameters")
        processing_layout = QFormLayout()
        processing_group.setLayout(processing_layout)

        # Processing Method
        self.processing_method_label = QLabel("Method:")
        self.radio_canny = QRadioButton("Canny Edge Detection")
        self.radio_threshold = QRadioButton("Binary Thresholding")
        self.radio_canny.setChecked(True) # Default
        processing_method_layout = QHBoxLayout()
        processing_method_layout.addWidget(self.radio_canny)
        processing_method_layout.addWidget(self.radio_threshold)
        processing_layout.addRow(self.processing_method_label, processing_method_layout)
        
        self.radio_canny.toggled.connect(self.update_processing_inputs_visibility)
        self.radio_threshold.toggled.connect(self.update_processing_inputs_visibility)


        # Canny Thresholds (visible for Canny)
        self.canny_threshold1_input = QLineEdit("50")
        self.canny_threshold1_input.setValidator(QIntValidator(0, 255))
        self.canny_threshold2_input = QLineEdit("150")
        self.canny_threshold2_input.setValidator(QIntValidator(0, 255))
        self.canny_threshold1_label = QLabel("Canny Threshold 1:")
        self.canny_threshold2_label = QLabel("Canny Threshold 2:")
        
        processing_layout.addRow(self.canny_threshold1_label, self.canny_threshold1_input)
        processing_layout.addRow(self.canny_threshold2_label, self.canny_threshold2_input)

        # Threshold Value (visible for Threshold)
        self.threshold_value_input = QLineEdit("127")
        self.threshold_value_input.setValidator(QIntValidator(0, 255))
        self.threshold_value_label = QLabel("Threshold Value:")
        processing_layout.addRow(self.threshold_value_label, self.threshold_value_input)

        # Blur Kernel Size
        self.blur_kernel_input = QLineEdit("5")
        self.blur_kernel_input.setValidator(QIntValidator(1, 99)) # Example range, will enforce odd
        processing_layout.addRow("Blur Kernel (Odd):", self.blur_kernel_input)

        main_layout.addWidget(processing_group)

        # --- Calibration Group ---
        calibration_group = QGroupBox("Calibration (Pixels per mm)")
        # Create a layout for the calibration_group FIRST
        calibration_main_layout = QVBoxLayout() # Using QVBoxLayout to stack the form layout and the projection group
        calibration_group.setLayout(calibration_main_layout) # Set this layout to the calibration_group

        calibration_form_layout = QFormLayout()

        # Part Pixels/mm
        self.part_pixels_per_mm_label = QLabel("Part Pixels/mm: N/A")
        self.part_set_scale_button = QPushButton("Set Part Scale")
        self.part_set_scale_button.clicked.connect(self.set_part_scale)
        calibration_form_layout.addRow(self.part_pixels_per_mm_label, self.part_set_scale_button)

        # Reference Pixels/mm (conditionally visible for image reference)
        self.reference_calibration_group = QGroupBox() # Nested group for conditional visibility
        reference_calibration_layout = QFormLayout(self.reference_calibration_group) # Layout directly on group box
        self.reference_pixels_per_mm_label = QLabel("Reference Pixels/mm: N/A")
        self.reference_set_scale_button = QPushButton("Set Reference Scale")
        self.reference_set_scale_button.clicked.connect(self.set_reference_scale)
        reference_calibration_layout.addRow(self.reference_pixels_per_mm_label, self.reference_set_scale_button)

        # Add the form layout (containing part and reference scale) to the main calibration layout
        calibration_main_layout.addLayout(calibration_form_layout)
        # Add the nested reference_calibration_group to the main calibration layout
        calibration_main_layout.addWidget(self.reference_calibration_group)


        # STL Projection Axis (conditionally visible for STL reference)
        self.projection_group = QGroupBox("STL Projection Axis")
        projection_layout = QHBoxLayout()
        self.radio_x = QRadioButton("X-axis")
        self.radio_y = QRadioButton("Y-axis")
        self.radio_z = QRadioButton("Z-axis")
        self.radio_z.setChecked(True) # Default
        projection_layout.addWidget(self.radio_x)
        projection_layout.addWidget(self.radio_y)
        projection_layout.addWidget(self.radio_z)
        self.projection_group.setLayout(projection_layout)

        # Add the projection_group to the main calibration layout
        calibration_main_layout.addWidget(self.projection_group) # Add to calibration group, will be hidden by default

        main_layout.addWidget(calibration_group) # Add the whole calibration group to the main window's layout


        # --- Control Buttons ---
        control_buttons_layout = QHBoxLayout()
        self.perform_analysis_button = QPushButton("Perform Analysis")
        self.perform_analysis_button.clicked.connect(self.perform_analysis)
        self.perform_analysis_button.setEnabled(False) # Disabled by default

        self.reset_button = QPushButton("Reset All")
        self.reset_button.clicked.connect(self.reset_ui_state)

        control_buttons_layout.addWidget(self.perform_analysis_button)
        control_buttons_layout.addWidget(self.reset_button)
        main_layout.addLayout(control_buttons_layout)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.progress_bar)

        # Connect signals for enabling/disabling analysis button
        self.part_path_input.textChanged.connect(self._check_enable_analysis_button)
        self.reference_path_input.textChanged.connect(self._check_enable_analysis_button)


        # Initial visibility update
        self.update_processing_inputs_visibility()

    def update_processing_inputs_visibility(self):
        is_canny = self.radio_canny.isChecked()
        self.canny_threshold1_label.setVisible(is_canny)
        self.canny_threshold1_input.setVisible(is_canny)
        self.canny_threshold2_label.setVisible(is_canny)
        self.canny_threshold2_input.setVisible(is_canny)

        is_threshold = self.radio_threshold.isChecked()
        self.threshold_value_label.setVisible(is_threshold)
        self.threshold_value_input.setVisible(is_threshold)

    def browse_part_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Part Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.part_path_input.setText(file_path)
            logger.info(f"Part image selected: {file_path}")
            self.part_pixels_per_mm_label.setText("Part Pixels/mm: N/A") # Reset calibration on new image
            self.part_set_scale_button.setEnabled(True) # Enable button once image is selected

    def browse_reference_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference File", "", "Image Files (*.jpg *.jpeg *.png *.bmp);;STL Files (*.stl)")
        if file_path:
            self.reference_path_input.setText(file_path)
            logger.info(f"Reference file selected: {file_path}")
            # Determine reference type based on extension
            if file_path.lower().endswith(('.stl')):
                self.reference_calibration_group.setVisible(False) # Hide image calibration
                self.projection_group.setVisible(True) # Show STL projection options
                self.reference_type = 'stl'
                self.reference_pixels_per_mm_label.setText("Reference Pixels/mm: N/A (from STL)") # Indicate STL doesn't use this directly
                self.reference_set_scale_button.setEnabled(False) # Disable set scale for STL
            else: # Assume image if not STL
                self.reference_calibration_group.setVisible(True) # Show image calibration
                self.projection_group.setVisible(False) # Hide STL projection options
                self.reference_type = 'image'
                self.reference_pixels_per_mm_label.setText("Reference Pixels/mm: N/A")
                self.reference_set_scale_button.setEnabled(True) # Enable button for image reference

    def set_part_scale(self):
        self._set_scale_for_image(self.part_path_input.text(), self.part_pixels_per_mm_label)

    def set_reference_scale(self):
        # Only allow setting reference scale if reference type is image
        if self.reference_type == 'image':
            self._set_scale_for_image(self.reference_path_input.text(), self.reference_pixels_per_mm_label)
        else:
            QMessageBox.warning(self, "Calibration", "Reference calibration is not applicable for STL files.")


    def _set_scale_for_image(self, image_path, label_to_update):
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", "Please select an image first.")
            return

        try:
            # Load the image to get its dimensions
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image.")
            
            height, width = img.shape[:2]

            # Convert to QPixmap for display in a dialog
            qpixmap_image = convert_np_to_qpixmap(img, QSize(600, 400)) # Scale for dialog display

            dialog = QDialog(self)
            dialog.setWindowTitle("Set Scale")
            dialog_layout = QVBoxLayout()
            dialog.setLayout(dialog_layout)

            image_label = QLabel()
            image_label.setPixmap(qpixmap_image)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setScaledContents(True) # Ensure pixmap scales to label size
            image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # Allow label to be resized by layout
            dialog_layout.addWidget(image_label)
            dialog_layout.addWidget(QLabel("Click and drag on the image to measure a known distance."))

            self.line_start = QPoint()
            self.line_end = QPoint()
            self.drawing = False
            self.current_image_pixmap = qpixmap_image # Store for painting

            # Override paintEvent for the image_label temporarily
            image_label.paintEvent = lambda event: self._paint_scale_line(event, image_label)
            image_label.mousePressEvent = lambda event: self._mouse_press_scale(event, image_label)
            image_label.mouseMoveEvent = lambda event: self._mouse_move_scale(event, image_label)
            image_label.mouseReleaseEvent = lambda event: self._mouse_release_scale(event, image_label, width, height, label_to_update, dialog)

            dialog.exec() # Show dialog modally

        except Exception as e:
            logger.error(f"Error in set_scale: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Could not set scale: {e}")

    def _paint_scale_line(self, event, label):
        super(type(label), label).paintEvent(event) # Call original paint event to draw pixmap
        if self.drawing or (not self.line_start.isNull() and not self.line_end.isNull()):
            painter = QPainter(label)
            painter.setPen(QPen(QColor("red"), 2))
            painter.drawLine(self.line_start, self.line_end)
            painter.end()

    def _mouse_press_scale(self, event, label):
        if event.button() == Qt.MouseButton.LeftButton:
            self.line_start = event.pos()
            self.line_end = event.pos()
            self.drawing = True
            label.update() # Repaint to show the starting point

    def _mouse_move_scale(self, event, label):
        if self.drawing:
            self.line_end = event.pos()
            label.update() # Repaint to show the line growing

    def _mouse_release_scale(self, event, label, original_img_width, original_img_height, label_to_update, dialog):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.line_end = event.pos()
            label.update() # Final repaint

            # Calculate actual pixel distance on the original image
            # Need to account for scaling of the image_label's pixmap
            pixmap_width = label.pixmap().width()
            pixmap_height = label.pixmap().height()

            scale_x = original_img_width / pixmap_width
            scale_y = original_img_height / pixmap_height

            p1_orig = QPoint(int(self.line_start.x() * scale_x), int(self.line_start.y() * scale_y))
            p2_orig = QPoint(int(self.line_end.x() * scale_x), int(self.line_end.y() * scale_y))

            # Corrected: Convert QPoint to QPointF for QLineF constructor
            pixel_distance = QLineF(QPointF(p1_orig), QPointF(p2_orig)).length()

            if pixel_distance > 0:
                known_distance_mm, ok = QInputDialog.getDouble(self, "Enter Known Distance",
                                                            f"Enter the known real-world distance (in mm) for the drawn line ({pixel_distance:.2f} pixels):",
                                                            1.0, 0.01, 10000.0, 2)
                if ok:
                    pixels_per_mm = pixel_distance / known_distance_mm
                    label_to_update.setText(f"{label_to_update.text().split(':')[0]}: {pixels_per_mm:.2f}")
                    QMessageBox.information(self, "Calibration Set", f"Calibration set: {pixels_per_mm:.2f} pixels/mm")
                    dialog.accept() # Close the dialog
                else:
                    QMessageBox.warning(self, "Input Cancelled", "Calibration input cancelled.")
            else:
                QMessageBox.warning(self, "No Line Drawn", "Please draw a line on the image to measure distance.")
            
            # Reset line points after calibration or cancellation
            self.line_start = QPoint()
            self.line_end = QPoint()
            label.update()


    def perform_analysis(self):
        part_image_path = self.part_path_input.text()
        reference_path = self.reference_path_input.text()

        if not part_image_path or not os.path.exists(part_image_path):
            QMessageBox.warning(self, "Missing File", "Please select a valid Part Image.")
            return
        if not reference_path or not os.path.exists(reference_path):
            QMessageBox.warning(self, "Missing File", "Please select a valid Reference File.")
            return

        # Get processing parameters
        processing_method = 'Canny' if self.radio_canny.isChecked() else 'Threshold'
        blur_kernel_str = self.blur_kernel_input.text()
        try:
            blur_kernel = int(blur_kernel_str)
            if blur_kernel % 2 == 0:
                QMessageBox.warning(self, "Invalid Input", "Blur Kernel Size must be an odd number.")
                return
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Blur Kernel Size must be an integer.")
            return

        canny_threshold1 = int(self.canny_threshold1_input.text())
        canny_threshold2 = int(self.canny_threshold2_input.text())
        threshold_value = int(self.threshold_value_input.text())

        # Get calibration values
        part_pixels_per_mm_str = self.part_pixels_per_mm_label.text().split(': ')[1]
        part_pixels_per_mm = float(part_pixels_per_mm_str) if part_pixels_per_mm_str != "N/A" else None

        reference_pixels_per_mm = None # Initialize to None
        if self.reference_type == 'image':
            reference_pixels_per_mm_str = self.reference_pixels_per_mm_label.text().split(': ')[1]
            # Corrected: Check against "N/A" for image type
            reference_pixels_per_mm = float(reference_pixels_per_mm_str) if reference_pixels_per_mm_str != "N/A" else None
        elif self.reference_type == 'stl':
            # For STL, reference_pixels_per_mm is derived internally or not needed in this explicit form
            # It's already set to None at the beginning of the function and will remain so for STL.
            pass


        projection_axis = None
        if self.reference_type == 'stl':
            if self.radio_x.isChecked():
                projection_axis = 'X'
            elif self.radio_y.isChecked():
                projection_axis = 'Y'
            elif self.radio_z.isChecked():
                projection_axis = 'Z'
            
            # If STL, check if part_pixels_per_mm is set
            if part_pixels_per_mm is None:
                QMessageBox.warning(self, "Missing Calibration", "Please set 'Part Pixels/mm' for STL comparison.")
                return
            
            # For STL, reference_pixels_per_mm should be None or derived internally by ImageProcessor
            reference_pixels_per_mm = None # Ensure it's not mistakenly carried over from image reference

        else: # Image reference
            if part_pixels_per_mm is None or reference_pixels_per_mm is None:
                QMessageBox.warning(self, "Missing Calibration", "Please set 'Part Pixels/mm' and 'Reference Pixels/mm'.")
                return

        # Create AnalysisConfig object
        config = AnalysisConfig(
            image_path=part_image_path,
            reference_path=reference_path,
            reference_type=self.reference_type,
            projection_axis=projection_axis,
            threshold_value=threshold_value,
            canny_threshold1=canny_threshold1,
            canny_threshold2=canny_threshold2,
            blur_kernel=blur_kernel,
            processing_method=processing_method,
            part_pixels_per_mm_value=part_pixels_per_mm,
            reference_pixels_per_mm_value=reference_pixels_per_mm
        )
        logger.info(f"Analysis Configuration: {config}")

        # Disable UI during analysis
        self.perform_analysis_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.browse_part_button.setEnabled(False)
        self.browse_reference_button.setEnabled(False)
        self.part_set_scale_button.setEnabled(False)
        self.reference_set_scale_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Progress: Starting analysis...")
        QApplication.processEvents() # Update UI immediately

        # Start analysis in a separate thread
        self.analysis_worker = AnalysisWorker(config)
        self.analysis_worker.analysis_completed.connect(self.analysis_finished)
        self.analysis_worker.analysis_error.connect(self.analysis_error)
        self.analysis_worker.progress_updated_with_text.connect(self.update_progress)
        self.analysis_worker.start()
        logger.info("Analysis worker started.")

    def update_progress(self, percentage, text):
        self.progress_bar.setValue(percentage)
        self.progress_bar.setFormat(f"Progress: {text} (%p%)")
        QApplication.processEvents() # Ensure UI updates

    def analysis_finished(self, results):
        logger.info("Analysis finished. Displaying results.")
        self.progress_bar.setFormat("Progress: Complete!")
        self.progress_bar.setValue(100)
        QApplication.processEvents()

        # Re-enable controls
        self.reset_button.setEnabled(True)
        self.browse_part_button.setEnabled(True)
        self.browse_reference_button.setEnabled(True)
        self.part_set_scale_button.setEnabled(True)
        self.reference_set_scale_button.setEnabled(True)
        self._check_enable_analysis_button() # Re-enable perform button if paths are valid

        # Display results in a new window
        if self.comparison_window:
            self.comparison_window.close()
            self.comparison_window = None
            
        self.comparison_window = BoundaryComparisonWindow(results, self.analysis_worker.config) # Pass config
        self.comparison_window.show()
        logger.info("BoundaryComparisonWindow displayed.")

    def analysis_error(self, message):
        logger.error(f"Analysis error: {message}")
        self.progress_bar.setFormat("Progress: Error!")
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Error", message)

        # Re-enable controls
        self.reset_button.setEnabled(True)
        self.browse_part_button.setEnabled(True)
        self.browse_reference_button.setEnabled(True)
        self.part_set_scale_button.setEnabled(True)
        self.reference_set_scale_button.setEnabled(True)
        self._check_enable_analysis_button() # Re-enable perform button if paths are valid

    def reset_ui_state(self):
        logger.info("Resetting UI state.")
        self.part_path_input.clear()
        self.reference_path_input.clear()
        self.part_pixels_per_mm_label.setText("Part Pixels/mm: N/A")
        self.part_set_scale_button.setEnabled(False) # Disable until image is chosen

        self.reference_type = 'image' # Reset to default
        self.radio_z.setChecked(True) # Default STL projection to Z
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
        # Removed the line below as generate_report_button is not a member of PartAnalyzerApp
        # self.generate_report_button.setEnabled(False) # Disable report button on reset

        # Re-enable perform analysis button only if both paths are selected
        # This will be handled by enabling logic in browse_part_image/browse_reference
        # For now, keep disabled until files are chosen.
        self.perform_analysis_button.setEnabled(False)

        # Connect slots to file path changes to enable perform_analysis_button
        self.part_path_input.textChanged.connect(self._check_enable_analysis_button)
        self.reference_path_input.textChanged.connect(self._check_enable_analysis_button)

    def _check_enable_analysis_button(self):
        # Enable perform analysis button only if both image paths are selected and valid
        if self.part_path_input.text() and self.reference_path_input.text() and \
           os.path.exists(self.part_path_input.text()) and os.path.exists(self.reference_path_input.text()):
            self.perform_analysis_button.setEnabled(True)
        else:
            self.perform_analysis_button.setEnabled(False)

    def closeEvent(self, event):
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.quit()
            self.analysis_worker.wait()
        if self.comparison_window:
            self.comparison_window.close()
        event.accept()