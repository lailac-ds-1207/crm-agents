"""
PDF generation utility for the CRM-Agent system.
Provides classes and functions to create professional PDF reports.
"""
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
from datetime import datetime
import tempfile
from fpdf import FPDF
import matplotlib.pyplot as plt
from PIL import Image
import io

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs(config.REPORTS_DIR, exist_ok=True)

# PDF page settings
PAGE_WIDTH = 210  # A4 width in mm
PAGE_HEIGHT = 297  # A4 height in mm
MARGIN = 10  # margin in mm

# Font settings
FONT_FAMILY_MAIN = 'NotoSansKR'
FONT_FAMILY_HEADING = 'NotoSansKR'
FONT_SIZE_TITLE = 24
FONT_SIZE_HEADING1 = 18
FONT_SIZE_HEADING2 = 16
FONT_SIZE_HEADING3 = 14
FONT_SIZE_NORMAL = 11
FONT_SIZE_SMALL = 9

# Color definitions (RGB)
COLOR_PRIMARY = (31, 119, 180)  # Blue
COLOR_SECONDARY = (255, 127, 14)  # Orange
COLOR_TEXT = (0, 0, 0)  # Black
COLOR_HEADING = (50, 50, 50)  # Dark gray
COLOR_SUBHEADING = (100, 100, 100)  # Medium gray
COLOR_BACKGROUND = (255, 255, 255)  # White
COLOR_ACCENT = (44, 160, 44)  # Green

class ReportPDF(FPDF):
    """Extended FPDF class with additional report-specific functionality."""
    
    def __init__(self, orientation='P', unit='mm', format='A4'):
        """
        Initialize the PDF document.
        
        Args:
            orientation: Page orientation ('P' for Portrait, 'L' for Landscape)
            unit: Unit of measurement ('mm', 'pt', 'cm', 'in')
            format: Page format ('A4', 'Letter', 'Legal', etc.)
        """
        super().__init__(orientation=orientation, unit=unit, format=format)
        
        # Windows 시스템 폰트 경로 예시
        # 보통 NotoSansKR에는 Italic이 따로 없기 때문에, Regular을 재사용
        self.add_font('NotoSansKR', '', r"C:/Windows/Fonts/NotoSansKR-VF.ttf", uni=True)
        self.add_font('NotoSansKR', 'B', r"C:/Windows/Fonts/NotoSansKR-VF.ttf", uni=True)  # Bold 대체
        self.add_font('NotoSansKR', 'I', r"C:/Windows/Fonts/NotoSansKR-VF.ttf", uni=True)  # Italic 대체
        self.add_font('NotoSansKR', 'BI', r"C:/Windows/Fonts/NotoSansKR-VF.ttf", uni=True) # BoldItalic 대체
        # 기본 폰트를 NotoSansKR로 변경
        self.set_font('NotoSansKR','', FONT_SIZE_NORMAL)
        
        # Initialize document properties
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(True, margin=MARGIN)
        
        # Set default font - using built-in fonts instead of custom fonts
        # self.set_font(FONT_FAMILY_MAIN, '', FONT_SIZE_NORMAL)
        
        # Initialize document
        self.alias_nb_pages()  # For page numbers
        self.add_page()
        
        # Set document properties
        self.set_title("CDP Analysis Report")
        self.set_author("AI CRM Agent")
        self.set_creator("CDP AI Agent System")
        
        # Track the current y position for content flow
        self.current_y = MARGIN
        
        # Track section level for proper formatting
        self.section_level = 0
    
    def header(self):
        """Add header to each page."""
        # Save current position
        current_y = self.get_y()
        
        # Set header font
        self.set_font(FONT_FAMILY_HEADING, 'I', FONT_SIZE_SMALL)
        self.set_text_color(*COLOR_SUBHEADING)
        
        # Add header text
        self.cell(0, 10, "CDP AI Agent Analysis Report", 0, 0, 'L')
        
        # Add date on the right
        self.cell(0, 10, datetime.now().strftime("%Y-%m-%d"), 0, 0, 'R')
        
        # Add a line
        self.line(MARGIN, 18, PAGE_WIDTH - MARGIN, 18)
        
        # Reset position for content
        self.set_y(current_y + 15)
        
        # Reset text color
        self.set_text_color(*COLOR_TEXT)
    
    def footer(self):
        """Add footer to each page."""
        # Set footer position
        self.set_y(-15)
        
        # Set footer font
        self.set_font(FONT_FAMILY_MAIN, 'I', FONT_SIZE_SMALL)
        self.set_text_color(*COLOR_SUBHEADING)
        
        # Add page number
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    def set_title(self, title):
        """
        Set the report title and add it to the first page.
        
        Args:
            title: Report title text
        """
        # Set title font
        self.set_font(FONT_FAMILY_HEADING, 'B', FONT_SIZE_TITLE)
        self.set_text_color(*COLOR_PRIMARY)
        
        # Add title
        self.cell(0, 15, title, 0, 1, 'C')
        self.ln(5)
        
        # Reset text color
        self.set_text_color(*COLOR_TEXT)
        
        # Update current y position
        self.current_y = self.get_y()
    
    def add_date(self):
        """Add the current date below the title."""
        # Set font
        self.set_font(FONT_FAMILY_MAIN, 'I', FONT_SIZE_NORMAL)
        self.set_text_color(*COLOR_SUBHEADING)
        
        # Add date
        self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'C')
        self.ln(5)
        
        # Reset text color
        self.set_text_color(*COLOR_TEXT)
        
        # Update current y position
        self.current_y = self.get_y()
    
    def add_section(self, title):
        """
        Add a main section heading.
        
        Args:
            title: Section title text
        """
        # Check if we need a page break
        if self.get_y() > PAGE_HEIGHT - 50:
            self.add_page()
        
        # Set section level
        self.section_level = 1
        
        # Set heading font
        self.set_font(FONT_FAMILY_HEADING, 'B', FONT_SIZE_HEADING1)
        self.set_text_color(*COLOR_PRIMARY)
        
        # Add section title
        self.ln(5)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
        
        # Add a line under the section title
        self.line(MARGIN, self.get_y(), PAGE_WIDTH - MARGIN, self.get_y())
        self.ln(5)
        
        # Reset text color
        self.set_text_color(*COLOR_TEXT)
        
        # Reset font to normal
        self.set_font(FONT_FAMILY_MAIN, '', FONT_SIZE_NORMAL)
        
        # Update current y position
        self.current_y = self.get_y()
    
    def add_subsection(self, title):
        """
        Add a subsection heading.
        
        Args:
            title: Subsection title text
        """
        # Check if we need a page break
        if self.get_y() > PAGE_HEIGHT - 40:
            self.add_page()
        
        # Set section level
        self.section_level = 2
        
        # Set heading font
        self.set_font(FONT_FAMILY_HEADING, 'B', FONT_SIZE_HEADING2)
        self.set_text_color(*COLOR_SECONDARY)
        
        # Add subsection title
        self.ln(3)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)
        
        # Reset text color
        self.set_text_color(*COLOR_TEXT)
        
        # Reset font to normal
        self.set_font(FONT_FAMILY_MAIN, '', FONT_SIZE_NORMAL)
        
        # Update current y position
        self.current_y = self.get_y()
    
    def add_subsubsection(self, title):
        """
        Add a sub-subsection heading.
        
        Args:
            title: Sub-subsection title text
        """
        # Check if we need a page break
        if self.get_y() > PAGE_HEIGHT - 30:
            self.add_page()
        
        # Set section level
        self.section_level = 3
        
        # Set heading font
        self.set_font(FONT_FAMILY_HEADING, 'B', FONT_SIZE_HEADING3)
        self.set_text_color(*COLOR_SUBHEADING)
        
        # Add subsubsection title
        self.ln(2)
        self.cell(0, 6, title, 0, 1, 'L')
        self.ln(1)
        
        # Reset text color
        self.set_text_color(*COLOR_TEXT)
        
        # Reset font to normal
        self.set_font(FONT_FAMILY_MAIN, '', FONT_SIZE_NORMAL)
        
        # Update current y position
        self.current_y = self.get_y()
    
    def add_text(self, text, style=''):
        """
        Add text paragraph to the document.
        
        Args:
            text: Text content
            style: Text style ('B' for bold, 'I' for italic, 'U' for underline, or combinations)
        """
        # Set font style
        self.set_font(FONT_FAMILY_MAIN, style, FONT_SIZE_NORMAL)
        
        # Add text with multi-cell for automatic wrapping
        self.multi_cell(0, 6, text)
        self.ln(3)
        
        # Reset font to normal
        self.set_font(FONT_FAMILY_MAIN, '', FONT_SIZE_NORMAL)
        
        # Update current y position
        self.current_y = self.get_y()
    
    def add_bullet_list(self, items):
        """
        Add a bullet point list.
        
        Args:
            items: List of text items
        """
        # Set font
        self.set_font(FONT_FAMILY_MAIN, '', FONT_SIZE_NORMAL)
        
        # Add each bullet point
        for item in items:
            # Check if we need a page break
            if self.get_y() > PAGE_HEIGHT - 20:
                self.add_page()
            
            # Add bullet point
            # Use simple ASCII dash instead of a Unicode bullet to ensure font support
            self.cell(5, 6, '-', 0, 0, 'L')
            self.multi_cell(0, 6, item)
            self.ln(1)
        
        self.ln(3)
        
        # Update current y position
        self.current_y = self.get_y()
    
    def add_numbered_list(self, items):
        """
        Add a numbered list.
        
        Args:
            items: List of text items
        """
        # Set font
        self.set_font(FONT_FAMILY_MAIN, '', FONT_SIZE_NORMAL)
        
        # Add each numbered item
        for i, item in enumerate(items, 1):
            # Check if we need a page break
            if self.get_y() > PAGE_HEIGHT - 20:
                self.add_page()
            
            # Add number
            self.cell(8, 6, f"{i}.", 0, 0, 'L')
            self.multi_cell(0, 6, item)
            self.ln(1)
        
        self.ln(3)
        
        # Update current y position
        self.current_y = self.get_y()
    
    def add_table(self, headers, data, col_widths=None):
        """
        Add a table to the document.
        
        Args:
            headers: List of column headers
            data: List of rows, each row is a list of cell values
            col_widths: List of column widths (if None, equal widths are used)
        """
        # Calculate column widths if not provided
        if col_widths is None:
            col_widths = [(PAGE_WIDTH - 2 * MARGIN) / len(headers)] * len(headers)
        
        # Check if table fits on current page
        table_height = 10 + 8 * (len(data) + 1)  # Estimate table height
        if self.get_y() + table_height > PAGE_HEIGHT - 20:
            self.add_page()
        
        # Set font for headers
        self.set_font(FONT_FAMILY_HEADING, 'B', FONT_SIZE_NORMAL)
        self.set_fill_color(*COLOR_PRIMARY)
        self.set_text_color(255, 255, 255)  # White text for header
        
        # Add headers
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 10, header, 1, 0, 'C', 1)
        self.ln()
        
        # Reset colors for data
        self.set_fill_color(240, 240, 240)  # Light gray for alternating rows
        self.set_text_color(*COLOR_TEXT)
        self.set_font(FONT_FAMILY_MAIN, '', FONT_SIZE_NORMAL)
        
        # Add data rows
        fill = False
        for row in data:
            # Check if we need a page break
            if self.get_y() > PAGE_HEIGHT - 20:
                self.add_page()
                
                # Repeat headers on new page
                self.set_font(FONT_FAMILY_HEADING, 'B', FONT_SIZE_NORMAL)
                self.set_fill_color(*COLOR_PRIMARY)
                self.set_text_color(255, 255, 255)
                for i, header in enumerate(headers):
                    self.cell(col_widths[i], 10, header, 1, 0, 'C', 1)
                self.ln()
                
                # Reset colors for data
                self.set_fill_color(240, 240, 240)
                self.set_text_color(*COLOR_TEXT)
                self.set_font(FONT_FAMILY_MAIN, '', FONT_SIZE_NORMAL)
            
            # Add row data
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 8, str(cell), 1, 0, 'L', fill)
            self.ln()
            fill = not fill  # Alternate row colors
        
        self.ln(5)
        
        # Update current y position
        self.current_y = self.get_y()
    
    def add_image(self, image_path, width=None, height=None, caption=None):
        """
        Add an image to the document.
        
        Args:
            image_path: Path to the image file
            width: Desired width (if None, uses actual image width up to page width)
            height: Desired height (if None, maintains aspect ratio)
            caption: Optional caption text
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return
            
            # Get image dimensions
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Calculate dimensions to fit on page
            max_width = PAGE_WIDTH - 2 * MARGIN
            
            if width is None:
                width = min(img_width, max_width)
            
            if height is None:
                # Maintain aspect ratio
                height = width * img_height / img_width
            
            # Check if image fits on current page
            total_height = height + (15 if caption else 5)
            if self.get_y() + total_height > PAGE_HEIGHT - 20:
                self.add_page()
            
            # Calculate x position to center the image
            x = MARGIN + (max_width - width) / 2
            
            # Add image
            self.image(image_path, x=x, y=self.get_y(), w=width, h=height)
            self.ln(height + 2)
            
            # Add caption if provided
            if caption:
                self.set_font(FONT_FAMILY_MAIN, 'I', FONT_SIZE_SMALL)
                self.cell(0, 5, caption, 0, 1, 'C')
                self.ln(5)
                self.set_font(FONT_FAMILY_MAIN, '', FONT_SIZE_NORMAL)
            else:
                self.ln(5)
            
            # Update current y position
            self.current_y = self.get_y()
            
        except Exception as e:
            logger.error(f"Error adding image {image_path}: {str(e)}")
    
    def add_chart(self, plt_figure, width=None, caption=None):
        """
        Add a matplotlib figure to the document.
        
        Args:
            plt_figure: Matplotlib figure object
            width: Desired width (if None, uses default width)
            caption: Optional caption text
        """
        try:
            # Save figure to a temporary buffer
            buf = io.BytesIO()
            plt_figure.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            buf.seek(0)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                tmp.write(buf.getvalue())
                tmp_path = tmp.name
            
            # Add the image
            self.add_image(tmp_path, width=width, caption=caption)
            
            # Clean up
            plt.close(plt_figure)
            os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Error adding chart: {str(e)}")
    
    def add_plotly_chart(self, plotly_fig, width=None, caption=None):
        """
        Add a plotly figure to the document.
        
        Args:
            plotly_fig: Plotly figure object
            width: Desired width (if None, uses default width)
            caption: Optional caption text
        """
        try:
            # Save figure to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                plotly_fig.write_image(tmp.name, format='png', width=1200, height=800)
                tmp_path = tmp.name
            
            # Add the image
            self.add_image(tmp_path, width=width, caption=caption)
            
            # Clean up
            os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Error adding plotly chart: {str(e)}")
    
    def add_page_break(self):
        """Add a page break."""
        self.add_page()
        self.current_y = self.get_y()
    
    def add_toc(self):
        """Add a table of contents placeholder (to be filled later)."""
        # This is a placeholder - TOC generation requires a two-pass approach
        # which is not implemented in this basic version
        self.add_section("Table of Contents")
        self.add_text("(Table of contents will be generated automatically)")
        self.ln(10)
    
    def add_executive_summary(self, summary_text):
        """
        Add an executive summary section.
        
        Args:
            summary_text: Executive summary text
        """
        self.add_section("Executive Summary")
        
        # Add a box around the summary
        self.set_fill_color(245, 245, 245)  # Light gray background
        self.set_draw_color(*COLOR_PRIMARY)  # Blue border
        
        # Calculate text height
        self.set_font(FONT_FAMILY_MAIN, 'I', FONT_SIZE_NORMAL)
        text_height = self.get_string_height(0, summary_text)
        
        # Draw the box
        self.rect(MARGIN, self.get_y(), PAGE_WIDTH - 2 * MARGIN, text_height + 10, 'DF')
        
        # Add the summary text
        self.set_xy(MARGIN + 5, self.get_y() + 5)
        self.multi_cell(PAGE_WIDTH - 2 * MARGIN - 10, 6, summary_text)
        self.ln(10)
        
        # Reset font and colors
        self.set_font(FONT_FAMILY_MAIN, '', FONT_SIZE_NORMAL)
        self.set_fill_color(255, 255, 255)
        self.set_draw_color(0, 0, 0)
        
        # Update current y position
        self.current_y = self.get_y()
    
    def get_string_height(self, width, text):
        """
        Calculate the height of a string when rendered with multi_cell.
        
        Args:
            width: Width of the cell
            text: Text to measure
            
        Returns:
            Estimated height in mm
        """
        # This is a rough estimate
        if width == 0:
            width = self.w - self.r_margin - self.l_margin
        
        lines = len(text.split('\n'))
        if width > 0:
            # Add lines due to text wrapping
            chars_per_line = int(width / (self.get_string_width('x') / self.k))
            if chars_per_line > 0:
                lines += len(text) // chars_per_line
        
        return lines * self.font_size * 0.5
    
    def output(self, dest=None):
        """
        Output the PDF to a file or return it as a bytes object.
        
        Args:
            dest: Destination file path (if None, generates one)
            
        Returns:
            Path to the saved file
        """
        if dest is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dest = os.path.join(config.REPORTS_DIR, f"report_{timestamp}.pdf")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        
        # Output PDF
        # Call the parent FPDF.output to avoid infinite recursion
        super().output(dest)
        logger.info(f"PDF report saved to {dest}")
        
        return dest


class ReportGenerator:
    """High-level class for generating reports with predefined templates."""
    
    def __init__(self):
        """Initialize the report generator."""
        pass
    
    def create_trend_analysis_report(
        self,
        title: str,
        executive_summary: str,
        trend_data: Dict[str, Any],
        visualizations: List[str],
        insights: List[str],
        recommendations: List[str]
    ) -> str:
        """
        Create a trend analysis report.
        
        Args:
            title: Report title
            executive_summary: Executive summary text
            trend_data: Dictionary of trend analysis data
            visualizations: List of paths to visualization images
            insights: List of insight texts
            recommendations: List of recommendation texts
            
        Returns:
            Path to the generated PDF report
        """
        # Create PDF document
        pdf = ReportPDF()
        pdf.set_title(title)
        pdf.add_date()
        
        # Add executive summary
        pdf.add_executive_summary(executive_summary)
        
        # Add trend analysis section
        pdf.add_section("Trend Analysis")
        
        # Add trend data and visualizations
        for key, data in trend_data.items():
            pdf.add_subsection(key)
            pdf.add_text(data['description'])
            
            # Add visualization if available
            if 'visualization' in data and data['visualization'] in visualizations:
                pdf.add_image(data['visualization'], width=160, caption=data['caption'])
        
        # Add insights section
        pdf.add_section("Key Insights")
        pdf.add_bullet_list(insights)
        
        # Add recommendations section
        pdf.add_section("Recommendations")
        pdf.add_numbered_list(recommendations)
        
        # Generate and save the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(config.REPORTS_DIR, f"trend_analysis_{timestamp}.pdf")
        pdf.output(report_path)
        
        return report_path
    
    def create_forecast_report(
        self,
        title: str,
        executive_summary: str,
        forecast_data: Dict[str, Any],
        visualizations: List[str],
        insights: List[str],
        recommendations: List[str]
    ) -> str:
        """
        Create a forecast report.
        
        Args:
            title: Report title
            executive_summary: Executive summary text
            forecast_data: Dictionary of forecast data
            visualizations: List of paths to visualization images
            insights: List of insight texts
            recommendations: List of recommendation texts
            
        Returns:
            Path to the generated PDF report
        """
        # Create PDF document
        pdf = ReportPDF()
        pdf.set_title(title)
        pdf.add_date()
        
        # Add executive summary
        pdf.add_executive_summary(executive_summary)
        
        # Add forecast section
        pdf.add_section("Demand Forecast")
        
        # Add forecast data and visualizations
        for key, data in forecast_data.items():
            pdf.add_subsection(key)
            pdf.add_text(data['description'])
            
            # Add visualization if available
            if 'visualization' in data and data['visualization'] in visualizations:
                pdf.add_image(data['visualization'], width=160, caption=data['caption'])
        
        # Add insights section
        pdf.add_section("Forecast Insights")
        pdf.add_bullet_list(insights)
        
        # Add recommendations section
        pdf.add_section("Recommended Actions")
        pdf.add_numbered_list(recommendations)
        
        # Generate and save the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(config.REPORTS_DIR, f"forecast_report_{timestamp}.pdf")
        pdf.output(report_path)
        
        return report_path
    
    def create_campaign_report(
        self,
        title: str,
        executive_summary: str,
        campaigns: List[Dict[str, Any]],
        strengths: List[Dict[str, Any]],
        weaknesses: List[Dict[str, Any]],
        discussion_summary: str
    ) -> str:
        """
        Create a campaign decision report.
        
        Args:
            title: Report title
            executive_summary: Executive summary text
            campaigns: List of campaign dictionaries
            strengths: List of business strength dictionaries
            weaknesses: List of business weakness dictionaries
            discussion_summary: Summary of the decision-making discussion
            
        Returns:
            Path to the generated PDF report
        """
        # Create PDF document
        pdf = ReportPDF()
        pdf.set_title(title)
        pdf.add_date()
        
        # Add executive summary
        pdf.add_executive_summary(executive_summary)
        
        # Add business analysis section
        pdf.add_section("Business Analysis")
        
        # Add strengths subsection
        pdf.add_subsection("Business Strengths")
        for strength in strengths:
            pdf.add_subsubsection(strength['description'])
            pdf.add_text(f"Related Categories: {', '.join(strength['related_categories'])}")
            pdf.add_text(f"Supporting Data: {strength['supporting_data']}")
            pdf.add_text(f"Business Opportunity: {strength['opportunity']}")
            pdf.ln(3)
        
        # Add weaknesses subsection
        pdf.add_subsection("Business Weaknesses")
        for weakness in weaknesses:
            pdf.add_subsubsection(weakness['description'])
            pdf.add_text(f"Related Categories: {', '.join(weakness['related_categories'])}")
            pdf.add_text(f"Supporting Data: {weakness['supporting_data']}")
            pdf.add_text(f"Mitigation Strategy: {weakness['mitigation_strategy']}")
            pdf.ln(3)
        
        # Add selected campaigns section
        pdf.add_section("Selected Campaigns")
        
        # Add each campaign
        for i, campaign in enumerate(campaigns, 1):
            pdf.add_subsection(f"{i}. {campaign['campaign_name']}")
            pdf.add_text(f"Type: {campaign['campaign_type']}")
            pdf.add_text(f"Target Categories: {', '.join(campaign['target_categories'])}")
            pdf.add_text(f"Description: {campaign['description']}")
            pdf.add_text(f"Rationale: {campaign['rationale']}")
            pdf.add_text(f"Expected Impact: {campaign['expected_impact']}")
            pdf.add_text(f"KPIs: {', '.join(campaign['kpis'])}")
            pdf.add_text(f"Priority Score: {campaign['priority_score']}/10")
            pdf.ln(5)
        
        # Add discussion summary section
        pdf.add_section("Decision Process")
        pdf.add_text(discussion_summary)
        
        # Generate and save the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(config.REPORTS_DIR, f"campaign_decision_{timestamp}.pdf")
        pdf.output(report_path)
        
        return report_path
    
    def create_segmentation_report(
        self,
        title: str,
        executive_summary: str,
        campaigns: List[Dict[str, Any]],
        segments: List[Dict[str, Any]]
    ) -> str:
        """
        Create a segmentation report.
        
        Args:
            title: Report title
            executive_summary: Executive summary text
            campaigns: List of campaign dictionaries
            segments: List of segment dictionaries with campaign mappings
            
        Returns:
            Path to the generated PDF report
        """
        # Create PDF document
        pdf = ReportPDF()
        pdf.set_title(title)
        pdf.add_date()
        
        # Add executive summary
        pdf.add_executive_summary(executive_summary)
        
        # Add segmentation overview section
        pdf.add_section("Segmentation Overview")
        pdf.add_text("This report defines and analyzes target segments for the selected campaigns.")
        
        # Add campaign-segment mapping table
        pdf.add_subsection("Campaign-Segment Mapping")
        
        # Create table data
        headers = ["Campaign", "Segment", "Customer Count", "Expected Response Rate"]
        data = []
        
        for segment in segments:
            campaign = next((c for c in campaigns if c['campaign_id'] == segment['campaign_id']), None)
            if campaign:
                data.append([
                    campaign['campaign_name'],
                    segment['segment_name'],
                    segment['customer_count'],
                    f"{segment['expected_response_rate']}%"
                ])
        
        # Add table
        pdf.add_table(headers, data)
        
        # Add detailed segments section
        pdf.add_section("Detailed Segment Analysis")
        
        # Add each segment
        for segment in segments:
            campaign = next((c for c in campaigns if c['campaign_id'] == segment['campaign_id']), None)
            if not campaign:
                continue
                
            pdf.add_subsection(f"{segment['segment_name']} (Campaign: {campaign['campaign_name']})")
            
            # Add segment definition
            pdf.add_subsubsection("Segment Definition")
            pdf.add_text(segment['segment_definition'])
            
            # Add SQL query
            pdf.add_subsubsection("SQL Query")
            pdf.add_text(segment['sql_query'])
            
            # Add segment statistics
            pdf.add_subsubsection("Segment Statistics")
            pdf.add_text(f"Customer Count: {segment['customer_count']}")
            pdf.add_text(f"Percentage of Total: {segment['percentage_of_total']}%")
            pdf.add_text(f"Expected Response Rate: {segment['expected_response_rate']}%")
            
            # Add segment characteristics
            if 'characteristics' in segment:
                pdf.add_subsubsection("Segment Characteristics")
                pdf.add_bullet_list(segment['characteristics'])
            
            # Add visualization if available
            if 'visualization' in segment:
                pdf.add_image(segment['visualization'], width=160, caption=f"{segment['segment_name']} Segment Analysis")
            
            pdf.ln(5)
        
        # Generate and save the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(config.REPORTS_DIR, f"segmentation_{timestamp}.pdf")
        pdf.output(report_path)
        
        return report_path
    
    def create_final_report(
        self,
        title: str,
        executive_summary: str,
        trend_summary: str,
        forecast_summary: str,
        campaigns: List[Dict[str, Any]],
        segments: List[Dict[str, Any]],
        recommendations: List[str],
        visualizations: List[str] = None
    ) -> str:
        """
        Create a comprehensive final report.
        
        Args:
            title: Report title
            executive_summary: Executive summary text
            trend_summary: Summary of trend analysis
            forecast_summary: Summary of forecast analysis
            campaigns: List of campaign dictionaries
            segments: List of segment dictionaries
            recommendations: List of recommendation texts
            visualizations: Optional list of paths to visualization images
            
        Returns:
            Path to the generated PDF report
        """
        # Create PDF document
        pdf = ReportPDF()
        pdf.set_title(title)
        pdf.add_date()
        
        # Add executive summary
        pdf.add_executive_summary(executive_summary)
        
        # Add trend analysis summary
        pdf.add_section("Trend Analysis Summary")
        pdf.add_text(trend_summary)
        
        # Add forecast summary
        pdf.add_section("Demand Forecast Summary")
        pdf.add_text(forecast_summary)
        
        # Add campaign summary
        pdf.add_section("Selected Campaign Summary")
        
        # Add each campaign
        for i, campaign in enumerate(campaigns, 1):
            pdf.add_subsection(f"{i}. {campaign['campaign_name']}")
            pdf.add_text(f"Type: {campaign['campaign_type']}")
            pdf.add_text(f"Description: {campaign['description']}")
            pdf.add_text(f"Expected Impact: {campaign['expected_impact']}")
            
            # Add related segment if available
            related_segments = [s for s in segments if s['campaign_id'] == campaign['campaign_id']]
            if related_segments:
                for segment in related_segments:
                    pdf.add_subsubsection(f"Target Segment: {segment['segment_name']}")
                    pdf.add_text(f"Customer Count: {segment['customer_count']}")
                    pdf.add_text(f"Expected Response Rate: {segment['expected_response_rate']}%")
            
            pdf.ln(3)
        
        # Add segment analysis section
        pdf.add_section("Segment Analysis Summary")
        
        # Create table data
        headers = ["Segment", "Customer Count", "Percentage", "Key Characteristic"]
        data = []
        
        for segment in segments:
            characteristics = segment.get('characteristics', [])
            main_char = characteristics[0] if characteristics else "N/A"
            data.append([
                segment['segment_name'],
                segment['customer_count'],
                f"{segment['percentage_of_total']}%",
                main_char
            ])
        
        # Add table
        pdf.add_table(headers, data)
        
        # Add visualizations if available
        if visualizations:
            pdf.add_section("Key Visualizations")
            for i, viz_path in enumerate(visualizations):
                if os.path.exists(viz_path):
                    pdf.add_image(viz_path, width=160, caption=f"Visualization {i+1}")
        
        # Add recommendations section
        pdf.add_section("Final Recommendations")
        pdf.add_numbered_list(recommendations)
        
        # Add next steps section
        pdf.add_section("Next Steps")
        pdf.add_text("Based on the analysis and recommendations in this report, please proceed with the following next steps:")
        pdf.add_bullet_list([
            "Develop campaign execution plan",
            "Create segment-specific marketing messages",
            "Set up campaign performance metrics",
            "Establish execution timeline and responsibilities",
            "Implement monitoring and feedback system"
        ])
        
        # Generate and save the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(config.REPORTS_DIR, f"final_report_{timestamp}.pdf")
        pdf.output(report_path)
        
        return report_path


# Initialize the report directory when module is imported
os.makedirs(config.REPORTS_DIR, exist_ok=True)
logger.info(f"Report directory setup at: {config.REPORTS_DIR}")
