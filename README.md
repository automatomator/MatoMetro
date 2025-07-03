# Part Analyzer Tool - Alpha Version 1 (Proof of Concept)

## Overview

This repository contains the initial alpha version of the Part Analyzer Tool, a proof-of-concept application designed for visual inspection and dimensional analysis of physical parts against a digital reference (either another image or a 3D STL model).

The tool aims to:
* Load an image of a physical part.
* Load a reference (image or STL) file.
* Automatically extract contours from images or project contours from STL models.
* Align the part contour to the reference contour using robust algorithms.
* Calculate and display key deviation metrics (e.g., area deviation, maximum point-to-point deviation).
* Provide visual feedback through superimposed images and a generated PDF report.

## Current Status (Alpha v1)

This version represents a working proof of concept with significant enhancements for robustness and functionality.

**Key Features Implemented:**
* **Modular Design:** Code is organized into `main`, `src/analysis_worker`, `src/image_processing`, `src/config_models`, `src/report_generator`, and `src/ui_windows` for better maintainability and scalability.
* **Image Loading:** Supports JPG and BMP for part and reference images.
* **Contour Extraction:** Uses configurable Gaussian blur and either Canny edge detection or binary thresholding.
* **Robust Alignment:** Employs centroid matching followed by `cv2.estimateAffine2D` (using RANSAC) for robust alignment of contours, handling translation, rotation, and scaling. Improved input handling for `estimateAffine2D` to prevent errors.
* **STL Reference Support:**
    * Ability to load `.stl` files as references.
    * Projects 3D STL models onto a 2D plane (X, Y, or Z axis configurable) and extracts their 2D convex hull as a reference contour.
    * Seamless integration into the analysis pipeline.
* **Deviation Metrics:** Calculates area deviation percentage and maximum point-to-point deviation (in mm, using calibration).
* **Visual Output:** Displays extracted part boundary, reference boundary, and a superimposed comparison.
* **Interactive Scale Calibration:** Allows users to define a known distance on an image to calculate pixels/mm for accurate dimensional analysis. This can be done for both part and reference images (if image-based).
* **Basic UI (PyQt6):** A functional graphical user interface for:
    * Selecting part and reference files.
    * Choosing reference type (image/STL) and STL projection axis.
    * Configuring image processing parameters (Canny thresholds, blur kernel, binary threshold).
    * Triggering the analysis.
    * Displaying progress and error messages.
    * Displaying detailed quantitative and visual results in a separate window.
* **Background Analysis:** Analysis runs in a separate QThread to prevent UI freezing.
* **PDF Report Generation:** Creates a comprehensive PDF report including analysis configuration, visual outputs, and quantitative metrics.
* **Robust Error Handling:** Improved logging and user-friendly error messages for common issues (e.g., file not found, no contours detected, STL processing errors).

## Project Roadmap (Next Phases)

**Phase 1: Core Functionality (Completed - Alpha v1)**
* Initial setup with basic image loading and contour detection.
* Basic alignment (e.g., centroid alignment).
* Initial deviation calculation.
* Basic UI for input and results display.
* **New:** Robust alignment with `cv2.estimateAffine2D`.
* **New:** STL reference support with 2D projection.
* **New:** Interactive scale calibration.
* **New:** Comprehensive PDF report generation.

**Phase 2: Refinement & Advanced Alignment (High Priority)**
* **Current:** Ensure `ConvexHull` is correctly imported and used where necessary.
* **Current:** Standardize `config` access throughout the modules (e.g., `config.attribute` instead of `config.get()`).
* **Current:** Verify robust behavior across various image and STL inputs for the alignment and deviation calculation pipeline.
* **Future:** Enable seamless switching between image-to-image and image-to-STL comparison.

**Phase 3: Enhanced Analysis & Visualization (Medium Priority)**
* **Future:** Implement localized deviation mapping (e.g., heatmap or color-coding on the superimposed image) to show where deviations are largest.
* **Future:** Add more sophisticated deviation metrics (e.g., Hausdorff distance, average absolute deviation).
* **Future:** Introduce tolerance zones for pass/fail criteria based on deviation metrics.

**Phase 4: Reporting & UX Improvements (Lower Priority)**
* Improve user input validation and provide hints for optimal parameter selection (e.g., real-time feedback on contour detection).
* Add save/load functionality for analysis configurations.
* Implement a persistent settings feature for UI parameters.

## Setup and Running

1.  **Prerequisites:**
    * Python 3.x
    * `pip` (Python package installer)

2.  **Install Dependencies:**
    Open your terminal or command prompt and navigate to your project's root directory (`C:/Users/singh/OneDrive/Desktop/logs/alpha_code/`). Then run:
    ```bash
    pip install opencv-python numpy PyQt6 scipy trimesh reportlab
    ```

3.  **Run the Application:**
    From the root directory of the project in your terminal:
    ```bash
    python main.py
    ```

---

This comprehensive set of codes should get your project fully updated with all the discussed optimizations and features. Once you've replaced these files, you can proceed with the Git workflow (check `git status`, `git add .`, `git commit -m "..."`, `git pull`, `git push`) to push these changes to your repository.