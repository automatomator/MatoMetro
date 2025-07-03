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

        # Configuration Details
        story.append(Paragraph("<b>Analysis Configuration:</b>", self.styles['h2']))
        story.append(Paragraph(f"Part Image: {os.path.basename(self.config.image_path)}", self.styles['Normal']))
        story.append(Paragraph(f"Reference Type: {self.config.reference_type.upper()}", self.styles['Normal']))
        story.append(Paragraph(f"Reference Path: {os.path.basename(self.config.reference_path)}", self.styles['Normal']))
        story.append(Paragraph(f"Processing Method: {self.config.processing_method}", self.styles['Normal']))
        story.append(Paragraph(f"Threshold Value: {self.config.threshold_value}", self.styles['Normal']))
        story.append(Paragraph(f"Blur Kernel Size: {self.config.blur_kernel}", self.styles['Normal']))
        
        # Display calibration values if available
        part_pixels_per_mm_display = f"{self.config.part_pixels_per_mm_value:.2f}" if self.config.part_pixels_per_mm_value is not None else "N/A (Pseudo-scaled)"
        reference_pixels_per_mm_display = f"{self.config.reference_pixels_per_mm_value:.2f}" if self.config.reference_pixels_per_mm_value is not None else "N/A (Pseudo-scaled)"
        
        story.append(Paragraph(f"Part Pixels/mm: {part_pixels_per_mm_display}", self.styles['Normal']))
        story.append(Paragraph(f"Reference Pixels/mm: {reference_pixels_per_mm_display}", self.styles['Normal']))

        if self.config.reference_type == 'stl':
            story.append(Paragraph(f"STL Projection Axis: {self.config.projection_axis}", self.styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Visual Outputs Section
        story.append(Paragraph("<b>Visual Outputs:</b>", self.styles['h2']))

        # Part Boundary Image
        # Check for 'part_processed_image' which is the drawn contour, not the original image
        if 'part_processed_image' in self.results and self.results['part_processed_image'] is not None:
            # Save the numpy image to a temporary file for ReportLab
            temp_part_boundary_path = os.path.join(tempfile.gettempdir(), "part_boundary_report.png")
            cv2.imwrite(temp_part_boundary_path, self.results['part_processed_image'])

            story.append(Paragraph("Processed Part Boundary:", self.styles['Normal']))
            story.append(Image(temp_part_boundary_path, width=3*inch, height=3*inch))
            story.append(Spacer(1, 0.1 * inch))
        elif 'part_boundary_image_path' in self.results and os.path.exists(self.results['part_boundary_image_path']):
             story.append(Paragraph("Processed Part Boundary:", self.styles['Normal']))
             story.append(Image(self.results['part_boundary_image_path'], width=3*inch, height=3*inch))
             story.append(Spacer(1, 0.1 * inch))
        else:
            story.append(Paragraph("Processed Part Boundary image not found for report.", self.styles['Normal']))

        story.append(Spacer(1, 0.2 * inch))

        # Reference Boundary Image
        if 'reference_boundary_image_path' in self.results and os.path.exists(self.results['reference_boundary_image_path']):
            story.append(Paragraph("Processed/Projected Reference Boundary:", self.styles['Normal']))
            story.append(Image(self.results['reference_boundary_image_path'], width=3*inch, height=3*inch))
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
        story.append(Paragraph(f"Image Area: {self.results.get('part_area_mm2', 0.0):.2f} mm²", self.styles['Normal']))
        story.append(Paragraph(f"Reference Area: {self.results.get('reference_area_mm2', 0.0):.2f} mm²", self.styles['Normal']))
        story.append(Paragraph(f"Area Deviation: {self.results.get('area_deviation_percent', 0.0):.2f} %", self.styles['Normal']))
        story.append(Paragraph(f"Maximum Contour Deviation: {self.results.get('max_deviation_mm', 0.0):.2f} mm", self.styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Build the PDF
        try:
            doc.build(story)
            print(f"Report generated successfully: {filename}")
        except Exception as e:
            print(f"Error generating PDF report: {e}")