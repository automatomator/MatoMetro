# Project Development Plan: Part Analyzer Tool

This document outlines the detailed short-term, mid-term, and long-term development goals for the Part Analyzer Tool.

---

## Current Date: Thursday, July 3, 2025

---

## Short-Term (Till Friday Night, July 4th) - **Critical for Report Submission**

**Primary Goal:** Successfully generate a PDF report that includes all processed images (Part, Reference/Projected STL, and Superimposed).

* **Task 1: Debug Image Embedding in PDF (High Priority)**
    * **Files to check:** `analysis_worker.py`, `report_generator.py`, and `ui_windows.py` (where results are passed).
    * **Action:**
        1.  In `analysis_worker.py`, after processing images (e.g., `part_processed_display_image`, `reference_processed_display_image`, `superimposed_image`), ensure these are **saved to temporary files** (e.g., using `cv2.imwrite` to paths in `tempfile.gettempdir()`). Make sure the format (e.g., PNG) is compatible with `reportlab`.
        2.  Store the *paths* to these saved images in the `results` dictionary that `analysis_completed` emits.
        3.  In `ui_windows.py`, when the `analysis_completed` signal is received, extract these image paths from the `results` dictionary and pass them to the `ReportGenerator` instance.
        4.  In `report_generator.py`, verify that `self.results['processed_part_image_path']`, `self.results['processed_reference_image_path']`, and `self.results['superimposed_image_path']` are correctly receiving valid, existing paths. Ensure the `Image()` constructor in `reportlab` is correctly referenced (it already is, but confirm its usage with the new paths).
    * **Expected Outcome:** A PDF report where "Processed Part Boundary image not found for report." and "Processed/Projected Reference Boundary image not found for report." are replaced by actual images. The superimposed image should also appear correctly.

* **Task 2: Populate Basic Report Fields (Low Priority if time is tight)**
    * **Files to check:** `ui_windows.py`, `report_generator.py`.
    * **Action:** Add simple input fields in the UI (e.g., `QLineEdit`s) for `Project Name` and `Operator Name` in `ui_windows.py`. Pass these values to the `generate_pdf_report` method of `ReportGenerator`.
    * **Expected Outcome:** The report includes basic project and operator information.

---

## Mid-Term (Weekend: July 5th - July 6th) - **Starting Interactive Adjustments**

**Primary Goal:** Implement initial mouse interaction for 2D image manipulation (pan and zoom).

* **Task 1: Enable 2D Image Pan (Translate)**
    * **Files to check:** `ui_windows.py` (specifically the `QLabel`s or custom widgets displaying `self.part_image_label` and `self.reference_image_label`).
    * **Action:**
        1.  Create custom `QLabel` subclasses or implement `mousePressEvent`, `mouseMoveEvent`, `mouseReleaseEvent` directly in `MainWindow` (or `PartAnalyzerApp`).
        2.  Store the initial mouse press position.
        3.  When the mouse is moved while pressed, calculate the delta and update the `QPixmap`'s position or the internal transformation matrix.
        4.  Redraw the image (e.g., by updating the `QPixmap` set on the label).
    * **Expected Outcome:** Users can click and drag the displayed part and reference images to pan them within their display areas.

* **Task 2: Implement 2D Image Zoom (Scale)**
    * **Files to check:** `ui_windows.py`.
    * **Action:**
        1.  Implement `wheelEvent` for the image display `QLabel`s.
        2.  Based on mouse wheel direction, scale the `QPixmap` (or the underlying transformation matrix).
        3.  Consider implementing zooming centered around the mouse cursor for better user experience.
    * **Expected Outcome:** Users can use the mouse wheel to zoom in and out of the part and reference images.

---

## Long-Term (Next Week: Starting July 7th) - **Expanding Features & Planning**

**Primary Goal:** Continue with interactive features (2D rotation, 3D manipulation) and begin architectural planning for new data types/analysis modes.

* **Task 1: Add 2D Image Rotation (Pre-analysis Adjustment)**
    * **Files to check:** `ui_windows.py`.
    * **Action:** Implement a mechanism (e.g., `Ctrl+Click` and drag, or a dedicated rotation tool/slider) to allow interactive rotation of the 2D images. This rotation should update an internal transformation matrix that is applied to the image *before* it's sent for contour detection and analysis.
    * **Expected Outcome:** Users can rotate the images for precise alignment before processing.

* **Task 2: Implement Basic 3D Model Interaction (for STL) (Pre-analysis Adjustment)**
    * **Files to check:** `ui_windows.py`, potentially a new 3D viewer module.
    * **Action:**
        1.  If you're displaying the STL as a simple 2D projection, the interaction (pan, zoom, rotate of the *projected view*) will be similar to 2D image manipulation. The goal here is to manipulate the 3D model's pose *before* projection.
        2.  Research suitable 3D viewer libraries for PyQt (e.g., integrating `PyOpenGL`, `VisPy`, or potentially a simpler `trimesh`/`PyQt` visualization if possible).
        3.  Start with basic camera controls (rotate, pan, zoom a 3D view of the STL). The crucial part is translating these 3D manipulations into the projection parameters (like `projection_axis` and offset) that affect the 2D contour used for analysis.
    * **Expected Outcome:** Users can interactively adjust the STL's orientation and scale in 3D before it's projected and used for analysis.

* **Task 3: Research and Conceptual Design for Assignment 2 (Point Cloud / CMM)**
    * **Action:**
        1.  Research Python libraries specifically for point cloud processing (e.g., `Open3D`, `PyVista`).
        2.  Familiarize yourself with common point cloud registration (alignment) algorithms (e.g., Iterative Closest Point for point clouds).
        3.  Sketch out how point cloud data would be loaded, visualized, and integrated into your existing analysis workflow or as a separate dedicated mode within the UI.
    * **Expected Outcome:** A clear conceptual understanding and a rough architectural plan for integrating point cloud data.

* **Task 4: Research and Conceptual Design for Assignment 3 (Propeller Analysis)**
    * **Action:**
        1.  Research methods for automatically extracting specific geometric parameters (diameter, pitch, number of blades, rotor end type) from 3D CAD models (e.g., using `trimesh`'s analytical capabilities or other geometric libraries).
        2.  Consider how to programmatically identify individual blades, leading/trailing edges, and measure pitch.
        3.  Sketch out the UI flow for a distinct "Propeller Analysis" mode.
    * **Expected Outcome:** A clear conceptual understanding of how to approach automated propeller analysis.