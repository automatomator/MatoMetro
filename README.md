# Part Analyzer Tool - Alpha Version 1 (Proof of Concept)

## Overview

This repository contains the initial alpha version of the Part Analyzer Tool, a proof-of-concept application designed for visual inspection and dimensional analysis of physical parts against a digital reference (currently, an image-based reference).

The tool aims to:
* Load an image of a physical part.
* Load a reference image (e.g., a CAD screenshot).
* Automatically extract contours from both images.
* Align the part contour to the reference contour using an ICP-like algorithm (Iterative Closest Point).
* Calculate and display key deviation metrics (e.g., area deviation, maximum point-to-point deviation).
* Provide visual feedback through superimposed images.

## Current Status (Alpha v1)

This version represents a working proof of concept.

**Key Features Implemented:**
* **Image Loading:** Supports JPG and BMP for part and reference images.
* **Contour Extraction:** Uses configurable Gaussian blur and either Canny edge detection or binary thresholding.
* **Robust Alignment:** Employs centroid matching followed by an ICP-like algorithm (`cv2.estimateAffine2D`) for combined translation, rotation, and scaling.
* **Deviation Metrics:** Calculates area deviation percentage and maximum point-to-point deviation (in pseudo-mm).
* **Visual Output:** Displays extracted part boundary, reference boundary, and a superimposed comparison.
* **Basic UI:** A functional PyQt6-based graphical user interface.

**Limitations / Known Issues (To be Addressed):**
1.  **Absolute Dimensional Accuracy (Critical):**
    * Current "mm" measurements are based on a "pseudo-scale" derived from the reference image's diagonal, assuming it represents an arbitrary 100mm.
    * **Issue:** This does not provide true, real-world dimensional accuracy.
    * **Solution (High Priority):** Implement a robust user-driven calibration feature where a known real-world distance can be measured on an input image (e.g., by clicking two points on a ruler or a calibrated object).
2.  **STL Reference Support (Partial):**
    * The framework for STL reference exists, but the 2D projection and alignment logic for STL models is not fully implemented or tested in this version. Currently, only image-to-image comparison is fully functional.
    * **Solution (Medium Priority):** Develop and integrate a robust 2D projection module for STL files.
3.  **Advanced Deviation Visualization:**
    * The current superimposed image shows overall alignment, but doesn't highlight specific areas of high deviation.
    * **Solution (Medium Priority):** Implement features to color-code or highlight regions where the part contour deviates significantly from the reference.
4.  **Reporting Functionality:**
    * No integrated reporting (e.g., PDF generation) of analysis results.
    * **Solution (Low Priority):** Add functionality to generate a comprehensive report including all metrics and visual comparisons.
5.  **Error Handling & User Feedback:**
    * While basic error handling is present, more user-friendly messages and guidance for parameter tuning could be added.

## Development Roadmap (Next Steps)

This section outlines the planned future development based on the current limitations and long-term vision.

**Phase 1: Core Accuracy & Calibration (High Priority)**
* Implement a "Calibration Mode" in the UI.
* Allow users to load a calibration image.
* Enable users to click two points on the image and input the real-world distance between them.
* Store and apply this `pixels_per_mm` scale factor consistently throughout the analysis.
* Refine contour extraction parameters and logic based on calibration insights.

**Phase 2: Full Reference Type Support (Medium Priority)**
* Develop robust 2D projection algorithms for STL models (e.g., orthographic projection along Z, X, Y axes).
* Integrate the STL projection into the alignment and deviation calculation pipeline.
* Enable seamless switching between image-to-image and image-to-STL comparison.

**Phase 3: Enhanced Analysis & Visualization (Medium Priority)**
* Implement localized deviation mapping (e.g., heatmap or color-coding on the superimposed image).
* Add more sophisticated deviation metrics (e.g., Hausdorff distance, average absolute deviation).
* Introduce tolerance zones for pass/fail criteria.

**Phase 4: Reporting & UX Improvements (Lower Priority)**
* Develop a customizable PDF report generation module.
* Improve user input validation and provide hints for optimal parameter selection.
* Add save/load functionality for analysis configurations.

## Setup and Running

1.  **Prerequisites:**
    * Python 3.x
    * `pip` (Python package installer)

2.  **Install Dependencies:**
    ```bash
    pip install opencv-python numpy PyQt6 scipy trimesh
    ```
    (Note: `trimesh` is currently only required if you plan to extend to STL support, but it's good to include for future work if you use it).

3.  **Run the Application:**
    Navigate to the root directory of the project in your terminal:
    ```bash
    cd C:/Users/singh/OneDrive/Desktop/logs/alpha_code/
    python main.py
    ```

---

This plan should help you structure your thoughts and future development effectively. Good luck!

For any questions contact mato: automatomator@gmail.com