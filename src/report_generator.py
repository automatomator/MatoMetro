# src/report_generator.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime
import os
import cv2

from .config_models import AnalysisConfig # Import the dataclass

class ReportGenerator:
    # Change config_data type hint
    def __init__(self, results_data, config_data: AnalysisConfig):
        self.results = results_data
        self.config = config_data # Store the dataclass instance
        self.styles = getSampleStyleSheet()

    def generate_pdf_report(self, filename="Part_Analysis_Report.pdf", project_name="", operator_name=""):
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []

        # Title
        story.append(Paragraph("<b>Part Deviation Analysis Report</b>", self.styles['h1']))
        story.append(Spacer(1, 0.2 * inch))

        # Date and User Info
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Date: {report_date}", self.styles['Normal']))
        if project_name:
            story.append(Paragraph(f"Project: {project_name}", self.styles['Normal']))
        if operator_name:
            story.append(Paragraph(f"Operator: {operator_name}", self.styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Analysis Configuration
        story.append(Paragraph("<b>Analysis Configuration:</b>", self.styles['h2']))
        # Access dataclass attributes directly
        story.append(Paragraph(f"Part Image: {os.path.basename(self.config.image_path)}", self.styles['Normal']))
        story.append(Paragraph(f"Reference Type: {self.config.reference_type.upper()}", self.styles['Normal']))
        story.append(Paragraph(f"Reference Path: {os.path.basename(self.config.reference_path)}", self.styles['Normal']))
        story.append(Paragraph(f"Processing Method: {self.config.processing_method}", self.styles['Normal']))
        
        if self.config.processing_method == 'Threshold':
            story.append(Paragraph(f"Threshold Value: {self.config.threshold_value}", self.styles['Normal']))
        else: # Canny
            story.append(Paragraph(f"Canny Threshold 1: {self.config.canny_threshold1}", self.styles['Normal']))
            story.append(Paragraph(f"Canny Threshold 2: {self.config.canny_threshold2}", self.styles['Normal']))
            
        story.append(Paragraph(f"Blur Kernel Size: {self.config.blur_kernel}", self.styles['Normal']))
        story.append(Paragraph(f"Part Pixels/mm: {self.results.get('part_pixels_per_mm', 'N/A'):.1f} (Used in Analysis)" if isinstance(self.results.get('part_pixels_per_mm'), (int, float)) else "Part Pixels/mm: N/A", self.styles['Normal']))
        story.append(Paragraph(f"Reference Pixels/mm: {self.results.get('reference_pixels_per_mm', 'N/A'):.1f} (Used in Analysis)" if isinstance(self.results.get('reference_pixels_per_mm'), (int, float)) else "Reference Pixels/mm: N/A", self.styles['Normal']))
        
        if self.config.reference_type == 'stl':
            story.append(Paragraph(f"STL Projection Axis: {self.config.projection_axis}", self.styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Visual Outputs
        story.append(Paragraph("<b>Visual Outputs:</b>", self.styles['h2']))

        # Part Boundary Image
        if 'part_contour_original_for_display' in self.results:
            # Save the numpy array to a temp image file for ReportLab
            temp_part_img_path = os.path.join(os.path.dirname(filename), "temp_part_boundary.png")
            cv2.imwrite(temp_part_img_path, self.results['part_contour_original_for_display'])
            story.append(Paragraph("Processed Part Boundary:", self.styles['Normal']))
            story.append(Image(temp_part_img_path, width=3*inch, height=3*inch))
            story.append(Spacer(1, 0.1 * inch))
        else:
            story.append(Paragraph("Processed Part Boundary image not found for report.", self.styles['Normal']))
        
        # Reference Boundary Image (could be image or STL projection)
        if 'reference_contour_original_for_display' in self.results:
            temp_ref_img_path = os.path.join(os.path.dirname(filename), "temp_ref_boundary.png")
            cv2.imwrite(temp_ref_img_path, self.results['reference_contour_original_for_display'])
            story.append(Paragraph("Processed/Projected Reference Boundary:", self.styles['Normal']))
            story.append(Image(temp_ref_img_path, width=3*inch, height=3*inch))
            story.append(Spacer(1, 0.1 * inch))
        else:
            story.append(Paragraph("Processed/Projected Reference Boundary image not found for report.", self.styles['Normal']))

        story.append(Spacer(1, 0.2 * inch))

        # Superimposed Image
        if 'superimposed_image_path' in self.results and os.path.exists(self.results['superimposed_image_path']):
            story.append(Paragraph("Superimposed (Aligned):", self.styles['Normal']))
            story.append(Image(self.results['superimposed_image_path'], width=3*inch, height=3*inch))
        else:
            story.append(Paragraph("Superimposed image not found for report.", self.styles['Normal']))

        story.append(Spacer(1, 0.2 * inch))

        # Quantitative Metrics
        story.append(Paragraph("<b>Quantitative Metrics:</b>", self.styles['h2']))
        story.append(Paragraph(f"Image Area: {self.results.get('image_area_mm2', 0.0):.2f} mm²", self.styles['Normal']))
        story.append(Paragraph(f"Reference Area: {self.results.get('reference_area_mm2', 0.0):.2f} mm²", self.styles['Normal']))
        story.append(Paragraph(f"Area Deviation: {self.results.get('area_deviation_percent', 0.0):.2f} %", self.styles['Normal']))
        story.append(Paragraph(f"Maximum Contour Deviation: {self.results.get('max_deviation_mm', 0.0):.2f} mm", self.styles['Normal']))
        
        story.append(Spacer(1, 0.2 * inch))

        doc.build(story)