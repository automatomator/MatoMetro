# Project Planning Document

## Current Status and Strategic Shift (July 03, 2025)

**Decision Point:** Following initial prototyping and a review of the current codebase's robustness and scalability, a strategic decision has been made to pivot the project direction. While core functional directives (Part vs. Reference comparison, contour extraction, etc.) remain unchanged, the primary focus for the next phase of development will be:

1.  **Enhanced Code Quality & Modularity:** Re-implementing foundational components with stricter adherence to software engineering best practices, including clearer separation of concerns, robust input validation, and comprehensive error handling.
2.  **Software Packaging Readiness:** Integrating modern Python packaging standards (`pyproject.toml`, `setuptools`) from the outset, aiming to create a distributable application (wheel) that can be easily installed and run. This will also serve as a learning exercise for future projects.
3.  **Improved Configuration Management:** Streamlining how analysis parameters are defined and potentially loaded/saved.
4.  **Temporary File Management:** Ensuring all temporary files generated during analysis (e.g., for reports) are handled cleanly and removed after use.

**Rationale:** This pivot is driven by the desire to build a more maintainable, scalable, and professional-grade application, and to acquire critical software packaging skills necessary for future projects.

**Immediate Next Steps (Before new private directory):**

1.  **Current Codebase Archival:** Push all current code and documentation (including `README.md`, `planning.md`, `log_book.docx`, and other relevant files) to the existing GitHub repository and **make it public**. This serves as a public archive of the proof-of-concept/alpha phase.
2.  **Define Program Requirements:** Before starting development in the new private directory, a detailed Program Requirements document will be drafted. This document will outline:
    * **Functional Requirements:** Specific features the application must provide.
    * **Non-Functional Requirements:** Performance, usability, security, maintainability, and packaging requirements.
    * **Technical Stack:** Confirmation of Python, PyQt6, OpenCV, NumPy, SciPy, Trimesh, ReportLab, and `build`/`setuptools`.
    * **Architectural Overview:** High-level design for modules and their interactions.

## Original Project Plan (Context/Previous Phases)

### Overview

This repository contains the initial alpha version of the Part Analyzer Tool, a proof-of-concept application designed for visual inspection and dimensional analysis of physical parts against a digital reference (currently, an image-based reference, with future plans for STL).

The tool aims to:
* Load an image of a physical part.
* Load a reference image (e.g., a CAD screenshot).
* Automatically extract contours from both images.
* Align the part contour to the reference contour using an ICP-like algorithm (Iterative Closest Point).
* Calculate and display key deviation metrics (e.g., area deviation, maximum point-to-point deviation).
* Provide visual feedback through superimposed images.

### Current Status (Alpha v1 - As of July 03, 2025 archival)

This version represents a working proof of concept.

**Key Features Implemented:**
* **Image Loading:** Supports JPG and BMP for part and reference images.
* **Contour Extraction:** Uses configurable Gaussian blur and either Canny edge detection or binary thresholding.
* **Robust Alignment:** Employs centroid matching followed by an ICP-like algorithm (`cv2.estimateAffine2D`) for combined translation, rotation, and scaling.
* **Deviation Metrics:** Calculates area deviation percentage and maximum point-to-point deviation (in pseudo-mm, with calibration conversion).
* **Visual Output:** Displays extracted part boundary, reference boundary, and a superimposed comparison.
* **Basic UI:** A functional PyQt6-based graphical user interface for user interaction and parameter input.
* **Background Processing:** Analysis runs in a separate thread to keep the UI responsive.
* **PDF Report Generation:** Basic PDF report generation functionality.
* **STL Projection:** Initial implementation for projecting STL models into 2D contours for comparison.

### Future Development Phases (Re-evaluation needed for new private directory)

*(Note: These phases will be re-evaluated and refined during the "Program Requirements" phase for the new private directory based on the new strategic direction.)*

**Phase 1: Core Functionality & Robustness (High Priority)**
* Establish robust image loading and preprocessing.
* Implement reliable contour extraction (e.g., handling noisy images, complex geometries).
* Develop a robust contour alignment and deviation calculation pipeline.
* **Future:** Enable seamless switching between image-to-image and image-to-STL comparison.

**Phase 2: Calibration & Accuracy (High Priority)**
* Refine the calibration mechanism for accurate real-world measurements.
* Validate calibration methods with physical objects and known dimensions.
* Implement robust error handling for all critical processing steps.

**Phase 3: Enhanced Analysis & Visualization (Medium Priority)**
* **Future:** Implement localized deviation mapping (e.g., heatmap or color-coding on the superimposed image).
* **Future:** Add more sophisticated deviation metrics (e.g., Hausdorff distance, average absolute deviation).
* **Future:** Introduce tolerance zones for pass/fail criteria.

**Phase 4: Reporting & UX Improvements (Lower Priority)**
* Develop a customizable PDF report generation module.
* Improve user input validation and provide hints for optimal parameter selection.
* Add save/load functionality for analysis configurations.

---

### Setup and Running (For the archived public version)

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
    cd C:/Users/singh/OneDrive/Desktop/logs/alpha_code/ # Adjust this path
    python main.py
    ```

---

*This planning document is a living artifact and will be updated as the project evolves.*