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
* **Basic UI:** A functional PyQt6-based graphical user interface for parameter input and result display.
* **STL Projection Framework:** Initial setup for projecting 3D STL models to 2D contours for comparison.
* **PDF Report Generation:** Basic PDF report for analysis results.

## Future Work / Roadmap

This section outlines the planned enhancements and new features. The **"Next Work Session Focus"** details immediate priorities.

### Phase 1: Core Functionality (Ongoing)
* Refine existing image processing and contour extraction for improved accuracy and robustness.
* Complete robust STL projection and deviation calculation, ensuring accurate area and maximum deviation metrics from 3D models.

### Phase 2: Robust Alignment (Ongoing)
* Further enhance alignment algorithms to handle a wider variety of part geometries and initial misalignments more effectively.

### Phase 3: Enhanced Analysis & Visualization (Medium Priority)
* Implement localized deviation mapping (e.g., heatmap or color-coding on the superimposed image) to visualize deviation distribution.
* Add more sophisticated deviation metrics (e.g., Hausdorff distance, average absolute deviation).
* Introduce tolerance zones for pass/fail criteria.

### Phase 4: Reporting & UX Improvements (Lower Priority)
* Develop a customizable PDF report generation module with more detailed visual outputs and configurable data.
* Improve user input validation and provide hints for optimal parameter selection.
* Add save/load functionality for analysis configurations.

### **Phase 5: Advanced Data Inputs & Interactive Pre-processing (Next Work Session Focus)**

This phase addresses the integration of new data types and enhanced user interaction capabilities.

1.  **Interactive Pre-analysis Adjustments:**
    * **2D Image Manipulation:** Implement features allowing users to interactively adjust scale, rotation, and translation of input images (both part and reference) using mouse gestures *before* the core analysis is performed. This includes:
        * Manual scaling/zoom of images.
        * Interactive rotation of images.
        * Panning/translation of images within the view.
        * Potential for basic cropping or masking of image regions.
    * **3D Model Manipulation (for STL References):** In the 3D mode, enable mouse interaction for initial scaling, rotation, and positioning of the loaded STL model for projection. This allows users to correctly orient and size the STL model for a proper 2D projection before it's used in the deviation analysis.

2.  **Point Cloud / CMM Data Integration:**
    * Develop the backend and UI components to handle and process 3D point cloud data or CMM (Coordinate Measuring Machine) scan data of manufactured parts.
    * Enable direct comparison of this 3D point cloud data with CAD models (STL format) or 2D orthographic drawing files.
    * Design an interactive UI/UX specifically for this point cloud workflow, allowing for visualization and analysis. (Note: Backend development for this is deferred, but UI/UX design can start).

3.  **Specialized Propeller Analysis Module:**
    * Introduce a distinct "Propeller Analysis" mode within the application.
    * Develop algorithms to automatically extract and compare key geometric parameters from two CAD models of propellers (e.g., one with serrations, one without). This comparison should verify if they share the same:
        * Overall Diameter
        * Blade Pitch
        * Number of Blades
        * Rotor End Type

4.  **Performance Evaluation & Design of Experiments (DOE) Framework:**
    * Develop a mechanism to programmatically change the resolution of input images (part and reference) for systematic testing.
    * Implement a framework to conduct a Design of Experiments (DOE), allowing for automated variation of parameters (like image resolution, blur kernel size, Canny thresholds, etc.) across multiple analysis runs.
    * Automate the collection, storage, and initial reporting of performance metrics (e.g., accuracy, processing time) from these DOE runs.

## Setup and Running

1.  **Prerequisites:**
    * Python 3.x
    * `pip` (Python package installer)

2.  **Install Dependencies:**
    ```bash
    pip install opencv-python numpy PyQt6 scipy trimesh reportlab
    ```
    (Note: `trimesh` is currently only required if you plan to extend to STL support, but it's good to include for future work if you use it).

3.  **Run the Application:**
    Navigate to the root directory of the project in your terminal:
    ```bash
    cd C:/Users/singh/OneDrive/Desktop/logs/alpha_code/
    python main.py
    ```

---