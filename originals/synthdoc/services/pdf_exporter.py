"""
PDF Export functionality for the document generator.
This module provides PDF export capabilities for generated HTML documents
and PDF-to-image conversion.
"""
import os
from pathlib import Path
from typing import List, Dict, Any


class PDFExporter:
    """
    PDF Export functionality using various backends and PDF-to-image conversion
    """
    def __init__(self):
        self.supported_engines = ["weasyprint", "playwright", "wkhtmltopdf"]
        self.engine = None
        self.pdf_to_image_engine = None
        self._detect_available_engine()
        self._detect_pdf_to_image_engine()
    
    def _detect_available_engine(self):
        """Detect which PDF generation engine is available"""
        try:
            import weasyprint
            # Try to initialize weasyprint to catch platform-specific issues
            self.engine = "weasyprint"
            return
        except (ImportError, OSError):
            pass

        try:
            from playwright.sync_api import sync_playwright
            self.engine = "playwright"
            return
        except ImportError:
            pass

        # Check for wkhtmltopdf command line tool
        import subprocess
        try:
            result = subprocess.run(["wkhtmltopdf", "--version"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.engine = "wkhtmltopdf"
                return
        except FileNotFoundError:
            pass

        # If no engine is available, we'll provide instructions
        self.engine = None

    def _detect_pdf_to_image_engine(self):
        """Detect which PDF-to-image conversion engine is available"""
        # Try PyMuPDF first as it's more reliable and doesn't require external dependencies
        try:
            import fitz  # PyMuPDF
            self.pdf_to_image_engine = "pymupdf"
            return
        except ImportError:
            pass

        try:
            import pdf2image
            # Try to verify poppler is available
            try:
                from pdf2image.exceptions import PDFInfoNotInstalledError
                # Test if pdfinfo works
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                    test_path = f.name
                try:
                    pdf2image.pdfinfo_from_path(test_path)
                except (PDFInfoNotInstalledError, FileNotFoundError):
                    # poppler not available, skip pdf2image
                    pass
                else:
                    self.pdf_to_image_engine = "pdf2image"
                    return
                finally:
                    import os
                    if os.path.exists(test_path):
                        os.unlink(test_path)
            except:
                # If test fails, skip pdf2image
                pass
        except ImportError:
            pass

        self.pdf_to_image_engine = None

    def convert_html_to_pdf(self, html_path: str, pdf_path: str = None) -> str:
        """
        Convert an HTML file to PDF using the best available engine
        """
        if not pdf_path:
            pdf_path = html_path.replace('.html', '.pdf')
        
        if self.engine == "weasyprint":
            return self._convert_with_weasyprint(html_path, pdf_path)
        elif self.engine == "playwright":
            return self._convert_with_playwright(html_path, pdf_path)
        elif self.engine == "wkhtmltopdf":
            return self._convert_with_wkhtmltopdf(html_path, pdf_path)
        else:
            raise Exception(
                "No PDF generation engine found. Please install one of: "
                "weasyprint, playwright, or ensure wkhtmltopdf is in your PATH."
            )
    
    def _convert_with_weasyprint(self, html_path: str, pdf_path: str) -> str:
        """Convert HTML to PDF using WeasyPrint"""
        import weasyprint
        
        html_doc = weasyprint.HTML(html_path)
        html_doc.write_pdf(pdf_path)
        return pdf_path
    
    def _convert_with_playwright(self, html_path: str, pdf_path: str) -> str:
        """Convert HTML to PDF using Playwright"""
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Use file:// protocol to load local HTML file
            page.goto(f"file://{os.path.abspath(html_path)}", wait_until="networkidle")
            
            # Generate PDF
            page.pdf(
                path=pdf_path,
                format="A4",
                print_background=True
            )
            
            browser.close()
        
        return pdf_path
    
    def _convert_with_wkhtmltopdf(self, html_path: str, pdf_path: str) -> str:
        """Convert HTML to PDF using wkhtmltopdf command line tool"""
        import subprocess

        cmd = ["wkhtmltopdf", html_path, pdf_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"wkhtmltopdf failed: {result.stderr}")

        return pdf_path

    def convert_pdf_to_images(self, pdf_path: str, output_dir: str = None,
                             dpi: int = 300, format: str = 'jpeg') -> List[str]:
        """
        Convert a PDF file to images (one image per page).

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images (defaults to same directory as PDF)
            dpi: Resolution for the images (default 300 for high quality)
            format: Image format ('jpeg' or 'png')

        Returns:
            List of paths to the generated images (one per page)
        """
        if self.pdf_to_image_engine == "pdf2image":
            return self._convert_with_pdf2image(pdf_path, output_dir, dpi, format)
        elif self.pdf_to_image_engine == "pymupdf":
            return self._convert_with_pymupdf(pdf_path, output_dir, dpi, format)
        else:
            raise Exception(
                "No PDF-to-image conversion engine found. Please install one of: "
                "pdf2image (pip install pdf2image) or PyMuPDF (pip install PyMuPDF)"
            )

    def _convert_with_pdf2image(self, pdf_path: str, output_dir: str = None,
                                dpi: int = 300, format: str = 'jpeg') -> List[str]:
        """Convert PDF to images using pdf2image"""
        from pdf2image import convert_from_path

        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get base filename
        base_name = Path(pdf_path).stem

        # Convert PDF to list of PIL Images
        images = convert_from_path(pdf_path, dpi=dpi)

        # Save each page as a separate image
        image_paths = []
        for i, image in enumerate(images):
            if len(images) == 1:
                # Single page - no page suffix
                image_filename = f"{base_name}.{format}"
            else:
                # Multiple pages - add page number
                image_filename = f"{base_name}_page_{i+1}.{format}"

            image_path = output_dir / image_filename

            if format.lower() == 'jpeg':
                image.save(str(image_path), 'JPEG', quality=95)
            else:
                image.save(str(image_path), format.upper())

            image_paths.append(str(image_path))

        return image_paths

    def _convert_with_pymupdf(self, pdf_path: str, output_dir: str = None,
                              dpi: int = 300, format: str = 'jpeg') -> List[str]:
        """Convert PDF to images using PyMuPDF (fitz)"""
        import fitz

        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure output directory exists and is writable
        if not output_dir.exists():
            raise Exception(f"Output directory does not exist: {output_dir}")
        if not output_dir.is_dir():
            raise Exception(f"Output path is not a directory: {output_dir}")

        # Get base filename
        base_name = Path(pdf_path).stem

        # Open PDF
        doc = fitz.open(pdf_path)

        # Calculate zoom factor for desired DPI (72 is default PDF DPI)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        # Convert each page to image
        image_paths = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)

            if len(doc) == 1:
                # Single page - no page suffix
                image_filename = f"{base_name}.{format}"
            else:
                # Multiple pages - add page number
                image_filename = f"{base_name}_page_{page_num+1}.{format}"

            image_path = output_dir / image_filename

            if format.lower() == 'jpeg':
                pix.save(str(image_path), 'jpeg', jpg_quality=95)
            else:
                pix.save(str(image_path), format.lower())

            image_paths.append(str(image_path))

        doc.close()
        return image_paths

    def convert_html_to_images(self, html_path: str, output_dir: str = None,
                               dpi: int = 300, format: str = 'jpeg',
                               keep_pdf: bool = False) -> Dict[str, Any]:
        """
        Convert HTML to images via PDF (HTML -> PDF -> Images).
        This is the complete pipeline for one document.

        Args:
            html_path: Path to the HTML file
            output_dir: Directory to save images (defaults to same directory as HTML)
            dpi: Resolution for the images (default 300 for high quality)
            format: Image format ('jpeg' or 'png')
            keep_pdf: Whether to keep the intermediate PDF file (default False)

        Returns:
            Dictionary with 'pdf_path' and 'image_paths' (list of image files, one per page)
        """
        if output_dir is None:
            output_dir = os.path.dirname(html_path)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Convert HTML to PDF
        base_name = Path(html_path).stem
        pdf_path = output_dir / f"{base_name}.pdf"
        self.convert_html_to_pdf(html_path, str(pdf_path))

        # Step 2: Convert PDF to images (one per page)
        image_paths = self.convert_pdf_to_images(str(pdf_path), str(output_dir), dpi, format)

        # Step 3: Optionally remove the PDF
        if not keep_pdf:
            os.remove(pdf_path)
            pdf_path = None

        return {
            'pdf_path': str(pdf_path) if pdf_path else None,
            'image_paths': image_paths
        }