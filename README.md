# Part Analyzer Tool - Alpha Version 1 (Proof of Concept)

## Overview

This repository contains the initial alpha version of the Part Analyzer Tool, a proof-of-concept application designed for visual inspection and dimensional analysis of physical parts against a digital reference.

The tool aims to:
* Load an image of a physical part.
* Load a reference (either another image or an STL 3D model).
* Automatically extract contours from both.
* Align the part contour to the reference contour using robust methods.
* Calculate and display key deviation metrics.
* Provide visual feedback through superimposed images and a detailed report.

## Current Status (Alpha v1)

This version represents a working proof of concept with the following key features:

**Key Features Implemented:**
* **Image Loading:** Supports JPG and BMP for part and reference images.
* **STL Reference Support:** Can load `.stl` 3D models as reference, projecting them to 2D contours along a specified axis (X, Y, or Z).
* **Contour Extraction:** Uses configurable Gaussian blur and either Canny edge detection or binary thresholding.
* **Robust Alignment:** Employs centroid matching followed by `cv2.estimateAffine2D` with RANSAC for combined translation, rotation, and scaling, providing robust alignment even with some outliers.
* **Deviation Metrics:** Calculates area deviation percentage and maximum point-to-point deviation (in pseudo-mm or calibrated mm).
* **Visual Output:** Displays extracted part boundary, reference boundary, and a superimposed comparison.
* **PDF Report Generation:** Creates a basic PDF report summarizing analysis results and including visual outputs.
* **Basic UI:** A functional PyQt6-based graphical user interface for easy interaction.

## Future Development Phases (Roadmap)

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
    pip install opencv-python numpy PyQt6 scipy trimesh reportlab
    ```

3.  **Run the Application:**
    Navigate to the root directory of the project in your terminal:
    ```bash
    cd C:/Users/singh/OneDrive/Desktop/logs/alpha_code/
    python main.py
    ```

---

This plan should help you structure your thoughts and future development effectively. Good luck!

For any questions contact... automatomator@gmail.com