"""
Paged.js Renderer - Page-aware HTML rendering using Paged.js polyfill

This module provides a clean library interface for rendering HTML with automatic
pagination using Paged.js, eliminating the need for manual page break calculation.

Usage:
    from synthdoc.pagedjs_renderer import PagedJSRenderer
    
    renderer = PagedJSRenderer()
    pages = renderer.render_and_extract(html_path, output_dir)
    
    for page in pages:
        print(f"Page {page['page_number']}: {len(page['annotations'])} annotations")
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from playwright.sync_api import sync_playwright, Page
import json
import shutil
import tempfile

# Import ontology mapper for dynamic type mappings
from synthdoc.ontology_mapper import get_ontology_mapper


class PagedJSRenderer:
    """
    Renders HTML documents with Paged.js pagination and extracts bounding boxes
    per page.
    
    This is a drop-in replacement for the manual page break calculation approach.
    """
    
    # Paged.js CDN URL (fallback)
    PAGEDJS_CDN = "https://unpkg.com/pagedjs/dist/paged.polyfill.js"
    # Local vendored copy (preferred)
    PAGEDJS_LOCAL = Path(__file__).parent / "vendor" / "paged.polyfill.js"

    # Default page dimensions (8.5" x 11" at ~141 DPI)
    DEFAULT_VIEWPORT_WIDTH = 1200
    DEFAULT_VIEWPORT_HEIGHT = 1555

    def __init__(
        self,
        viewport_width: int = DEFAULT_VIEWPORT_WIDTH,
        viewport_height: int = DEFAULT_VIEWPORT_HEIGHT,
        headless: bool = True,
        timeout: int = 10000
    ):
        """
        Initialize the Paged.js renderer.
        
        Args:
            viewport_width: Browser viewport width in pixels
            viewport_height: Browser viewport height in pixels  
            headless: Run browser in headless mode
            timeout: Timeout for Paged.js pagination in milliseconds
        """
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.headless = headless
        self.timeout = timeout
    
    def inject_pagedjs_script(self, html_content: str) -> str:
        """
        Inject Paged.js script into HTML if not already present.
        Also strips local <link rel="stylesheet"> tags since CSS is already
        inlined by get_template_string(). These dangling links cause
        ERR_FILE_NOT_FOUND and CORS errors when loaded from temp files.

        Args:
            html_content: Original HTML content

        Returns:
            HTML with Paged.js script injected and local CSS links removed
        """
        # Strip local stylesheet <link> tags (CSS is already inlined via <style>)
        # Matches <link ... href="styles.css" ...> and similar local refs
        import re
        html_content = re.sub(
            r'<link\s+[^>]*href=["\'][^"\':/]+\.css["\'][^>]*/?>',
            '', html_content
        )

        # Check if Paged.js is already included
        if 'pagedjs' in html_content.lower() or 'paged.polyfill' in html_content:
            return html_content

        # Use local vendored copy if available, otherwise fall back to CDN
        if self.PAGEDJS_LOCAL.exists():
            src = f'file:///{self.PAGEDJS_LOCAL.resolve().as_posix()}'
        else:
            src = self.PAGEDJS_CDN

        script_tag = f'<script src="{src}"></script>'

        if '</head>' in html_content:
            html_content = html_content.replace('</head>', f'{script_tag}\n</head>')
        else:
            html_content = f'<!DOCTYPE html><html><head>{script_tag}</head><body>{html_content}</body></html>'

        return html_content
    
    def ensure_page_css(self, html_content: str) -> str:
        """
        Ensure HTML has @page CSS rules for proper pagination.
        
        Args:
            html_content: HTML content
            
        Returns:
            HTML with @page CSS if not already present
        """
        # Check if @page rules exist - if so, don't override (template knows best)
        if '@page' in html_content:
            # Just add table header fix if not present
            if 'table-header-group' not in html_content:
                table_css = """
<style>
/* Ensure table headers repeat on each page */
thead {
    display: table-header-group;
}

/* Prevent orphaned table rows */
tbody tr {
    page-break-inside: avoid;
}
</style>
"""
                if '</head>' in html_content:
                    html_content = html_content.replace('</head>', f'{table_css}\n</head>')
            return html_content
        
        # Add default @page rules only if template doesn't have them
        page_css = """
<style>
@page {
    size: 8.5in 11in;
    margin: 0.75in;
}

/* Ensure table headers repeat on each page */
thead {
    display: table-header-group;
}

/* Prevent orphaned table rows */
tbody tr {
    page-break-inside: avoid;
}
</style>
"""
        
        if '</head>' in html_content:
            html_content = html_content.replace('</head>', f'{page_css}\n</head>')
        elif '<head>' in html_content:
            html_content = html_content.replace('<head>', f'<head>\n{page_css}')
        
        return html_content
    
    def wait_for_pagination(self, page: Page) -> int:
        """
        Wait for Paged.js to complete pagination.
        
        Args:
            page: Playwright page instance
            
        Returns:
            Number of pages created
            
        Raises:
            TimeoutError: If pagination doesn't complete within timeout
        """
        try:
            # Wait for Paged.js to create the pages container
            page.wait_for_selector('.pagedjs_pages', timeout=self.timeout)

            # Give Paged.js a moment to finish rendering
            page.wait_for_timeout(500)

            # Count pages
            page_count = page.evaluate('document.querySelectorAll(".pagedjs_page").length')

            return page_count
        except Exception as e:
            # Check if Paged.js script even loaded
            pagedjs_loaded = page.evaluate('typeof window.PagedPolyfill !== "undefined" || typeof Paged !== "undefined"')
            if not pagedjs_loaded:
                raise TimeoutError(
                    f"Paged.js script failed to load (check network/local file). "
                    f"Original error: {e}"
                )
            raise TimeoutError(f"Paged.js pagination timed out: {e}")
    
    def extract_page_data(self, page: Page) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes and metadata from all paginated pages.
        Uses RDFa attributes with ontology-based type mapping instead of hard-coded mappings.

        Args:
            page: Playwright page instance (after pagination)

        Returns:
            List of page data dictionaries with annotations
        """
        # Get ontology mapper and generate type mappings
        mapper = get_ontology_mapper()
        ontology_config = mapper.generate_javascript_config()
        
        # Inject the ontology config and use it in the extraction
        # Use string concatenation to avoid f-string brace escaping issues
        js_code = """() => {
            """ + ontology_config + """
            
            const pages = document.querySelectorAll('.pagedjs_page');

            return Array.from(pages).map((pageEl, pageIdx) => {
                const pageRect = pageEl.getBoundingClientRect();

                // Extract sections using DOCO ontology
                // Look for elements with typeof containing "doco:" or "doco-fd:" followed by "Section"
                const sections = Array.from(pageEl.querySelectorAll('[typeof*="Section"]')).filter(el => {
                    const typeofAttr = el.getAttribute('typeof') || '';
                    return typeofAttr.includes('doco:') || typeofAttr.includes('doco-fd:');
                }).map(el => {
                    const rect = el.getBoundingClientRect();

                    // Extract section type from typeof
                    const typeofAttr = el.getAttribute('typeof') || '';
                    const types = typeofAttr.split(/\\s+/);
                    let sectionType = 'section';

                    for (const type of types) {
                        if (type.startsWith('doco:') || type.startsWith('doco-fd:')) {
                            sectionType = type.replace(/^doco(-fd)?:/, '');
                            break;
                        }
                    }

                    // Extract section ID from resource attribute
                    const resourceAttr = el.getAttribute('resource') || '';
                    const sectionId = resourceAttr.replace(/^#/, '') || `section_${pageIdx}_${Math.random().toString(36).substr(2, 9)}`;

                    return {
                        section_id: sectionId,
                        section_type: sectionType,
                        rdfa_typeof: el.getAttribute('typeof') || null,
                        x: rect.left - pageRect.left,
                        y: rect.top - pageRect.top,
                        width: rect.width,
                        height: rect.height
                    };
                });

                // Extract fields using DOCO ontology
                // Look for elements with typeof containing "doco:" or "doco-fd:" followed by "Field"
                const fields = Array.from(pageEl.querySelectorAll('[typeof*="Field"]')).filter(el => {
                    const typeofAttr = el.getAttribute('typeof') || '';
                    return typeofAttr.includes('doco:') || typeofAttr.includes('doco-fd:');
                }).map(el => {
                    const rect = el.getBoundingClientRect();

                    // Extract field type from typeof
                    const typeofAttr = el.getAttribute('typeof') || '';
                    const types = typeofAttr.split(/\\s+/);
                    let fieldType = 'field';

                    for (const type of types) {
                        if (type.startsWith('doco:') || type.startsWith('doco-fd:')) {
                            fieldType = type.replace(/^doco(-fd)?:/, '');
                            break;
                        }
                    }

                    // Extract field ID from resource attribute
                    const resourceAttr = el.getAttribute('resource') || '';
                    const fieldId = resourceAttr.replace(/^#/, '') || `field_${pageIdx}_${Math.random().toString(36).substr(2, 9)}`;

                    // Find parent section
                    let parentSection = el.closest('[typeof*="Section"]');
                    if (parentSection) {
                        const typeofAttr = parentSection.getAttribute('typeof') || '';
                        if (!typeofAttr.includes('doco:') && !typeofAttr.includes('doco-fd:')) {
                            parentSection = null;
                        }
                    }
                    const parentSectionId = parentSection ? (parentSection.getAttribute('resource') || '').replace(/^#/, '') : null;

                    return {
                        field_id: fieldId,
                        field_type: fieldType,
                        section_id: parentSectionId,
                        rdfa_typeof: el.getAttribute('typeof') || null,
                        x: rect.left - pageRect.left,
                        y: rect.top - pageRect.top,
                        width: rect.width,
                        height: rect.height
                    };
                });

                // Extract ALL DOCO ontology elements for complete document structure
                // Include semantic values, structural elements, and table components
                const valueSelectors = [
                    // Values (semantic content) - include both doco: and doco-fd: namespaces
                    '[property*="doco:fieldValue"]',
                    '[property*="doco-fd:fieldValue"]',
                    '[property*="doco:cellValue"]',
                    '[property*="doco-fd:cellValue"]',
                    // Structural labels and headers
                    '[property*="doco:fieldLabel"]',
                    '[property*="doco-fd:fieldLabel"]',
                    '[property*="doco:columnHeader"]',
                    '[property*="doco-fd:columnHeader"]',
                    '[property*="doco:rowHeader"]',
                    '[property*="doco-fd:rowHeader"]',
                    '[property*="doco:sectionTitle"]',
                    '[property*="doco:title"]',
                    '[property*="doco:caption"]',
                    // Table cells (if not already captured)
                    '[property*="doco:tableCell"]',
                    '[property*="doco-fd:tableCell"]',
                    // Schema.org semantic properties (catch elements without doco-fd:fieldValue)
                    '[property*="schema:"]'
                ];

                const allValueElements = [...new Set(valueSelectors.flatMap(selector =>
                    Array.from(pageEl.querySelectorAll(selector))
                ))];

                const boxes = allValueElements.flatMap(el => {
                    const propertyAttr = el.getAttribute('property') || '';
                    const elRect = el.getBoundingClientRect();

                    // Check if this is a table cell value
                    const isCellValue = propertyAttr.includes('doco:cellValue') || propertyAttr.includes('doco-fd:cellValue') ||
                        propertyAttr.includes('doco:tableCell') || propertyAttr.includes('doco-fd:tableCell');

                    // For table cells, use the <td>/<th> bounding box for the structural annotation
                    let cellRect = elRect;
                    if (isCellValue) {
                        const tableCell = el.closest('td, th');
                        if (tableCell) {
                            cellRect = tableCell.getBoundingClientRect();
                        }
                    }

                    // Resolve column header for table cells
                    let columnHeader = null;
                    if (isCellValue) {
                        const td = el.closest('td, th');
                        const row = td ? td.closest('tr') : null;
                        const table = el.closest('table');
                        if (td && row && table) {
                            const cellIndex = Array.from(row.children).indexOf(td);
                            const thead = table.querySelector('thead');
                            if (thead) {
                                const headerRow = thead.querySelector('tr');
                                if (headerRow && headerRow.children[cellIndex]) {
                                    columnHeader = headerRow.children[cellIndex].textContent.trim();
                                }
                            }
                        }
                    }

                    // Find parent field and section using RDFa
                    let parentField = el.closest('[typeof*="Field"]');
                    if (parentField) {
                        const typeofAttr = parentField.getAttribute('typeof') || '';
                        if (!typeofAttr.includes('doco:') && !typeofAttr.includes('doco-fd:')) {
                            parentField = null;
                        }
                    }

                    let parentSection = el.closest('[typeof*="Section"]');
                    if (parentSection) {
                        const typeofAttr = parentSection.getAttribute('typeof') || '';
                        if (!typeofAttr.includes('doco:') && !typeofAttr.includes('doco-fd:')) {
                            parentSection = null;
                        }
                    }

                    const fieldId = parentField ? (parentField.getAttribute('resource') || '').replace(/^#/, '') : null;
                    const sectionId = parentSection ? (parentSection.getAttribute('resource') || '').replace(/^#/, '') : null;

                    // Fallback section_id: use parent table resource if no section found
                    let effectiveSectionId = sectionId;
                    if (!effectiveSectionId) {
                        const parentTable = el.closest('table[resource]');
                        if (parentTable) {
                            effectiveSectionId = (parentTable.getAttribute('resource') || '').replace(/^#/, '');
                        }
                    }

                    // Generate field_id for table cells from table + column header
                    let effectiveFieldId = fieldId;
                    if (!effectiveFieldId && columnHeader) {
                        const tableEl = el.closest('table[resource]');
                        const tableId = tableEl ? (tableEl.getAttribute('resource') || '').replace(/^#/, '') : 'table';
                        const headerSlug = columnHeader.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/_+$/, '');
                        effectiveFieldId = tableId + '_' + headerSlug;
                    }

                    // For spans nested inside table cells, inherit table context
                    if (!isCellValue) {
                        const parentCell = el.closest('td[property*="cellValue"], th[property*="cellValue"]');
                        if (parentCell) {
                            // Resolve column header from parent cell
                            if (!columnHeader) {
                                const row = parentCell.closest('tr');
                                const table = parentCell.closest('table');
                                if (row && table) {
                                    const cellIndex = Array.from(row.children).indexOf(parentCell);
                                    const thead = table.querySelector('thead');
                                    if (thead) {
                                        const headerRow = thead.querySelector('tr');
                                        if (headerRow && headerRow.children[cellIndex]) {
                                            columnHeader = headerRow.children[cellIndex].textContent.trim();
                                        }
                                    }
                                }
                            }
                            // Inherit section_id from parent cell context
                            if (!effectiveSectionId) {
                                let ps = parentCell.closest('[typeof*="Section"]');
                                if (ps) {
                                    const ta = ps.getAttribute('typeof') || '';
                                    if (ta.includes('doco:') || ta.includes('doco-fd:')) {
                                        effectiveSectionId = (ps.getAttribute('resource') || '').replace(/^#/, '');
                                    }
                                }
                                if (!effectiveSectionId) {
                                    const pt = parentCell.closest('table[resource]');
                                    if (pt) effectiveSectionId = (pt.getAttribute('resource') || '').replace(/^#/, '');
                                }
                            }
                            // Generate field_id from table + column header
                            if (!effectiveFieldId && columnHeader) {
                                const tableEl = parentCell.closest('table[resource]');
                                const tableId = tableEl ? (tableEl.getAttribute('resource') || '').replace(/^#/, '') : 'table';
                                const headerSlug = columnHeader.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/_+$/, '');
                                effectiveFieldId = tableId + '_' + headerSlug;
                            }
                        }
                    }

                    // Resolve table and row IDs for graph structure
                    let tableId = null;
                    let rowId = null;
                    {
                        // Use el.closest('table') then check for resource (Paged.js
                        // may clone tables without preserving all attributes)
                        const tbl = el.closest('table');
                        if (tbl) {
                            const res = (tbl.getAttribute('resource') || '').replace(/^#/, '');
                            if (res) {
                                tableId = res;
                            } else {
                                // Fallback: use typeof + position index
                                const allTables = Array.from(document.querySelectorAll('.pagedjs_page table'));
                                tableId = 'table_' + allTables.indexOf(tbl);
                            }
                            const rowEl = el.closest('tr');
                            if (rowEl) {
                                rowId = (rowEl.getAttribute('resource') || '').replace(/^#/, '') || null;
                                if (!rowId) {
                                    const container = rowEl.closest('thead, tbody, tfoot');
                                    if (container) {
                                        const prefix = container.tagName.toLowerCase();
                                        const idx = Array.from(container.children).filter(c => c.tagName === 'TR').indexOf(rowEl);
                                        rowId = tableId + '_' + prefix + '_' + idx;
                                    }
                                }
                            }
                        }
                    }

                    // Resolve semantic type (Schema.org typeof → property → parent typeof)
                    function resolveSemanticType(element) {{
                        let semType = getSchemaType(element);
                        if (!semType) {{
                            const propName = getPropertyName(element);
                            if (ONTOLOGY_CONFIG.subjectNamedProperties.includes(propName)) {{
                                return inferEntityTypeFromSubject(element);
                            }}
                            semType = inferEntityTypeFromProperty(propName);
                        }}
                        if (!semType || semType === 'TEXT') {{
                            let parent = element.parentElement;
                            let depth = 0;
                            while (parent && depth < 3) {{
                                const parentType = getSchemaType(parent);
                                if (parentType) {{
                                    semType = parentType;
                                    break;
                                }}
                                parent = parent.parentElement;
                                depth++;
                            }}
                        }}
                        return (semType && semType !== 'TEXT') ? semType : null;
                    }}

                    // Determine entity type using ontology-based mappings
                    const structuralType = mapStructuralProperty(propertyAttr);
                    const entityType = structuralType || resolveSemanticType(el) || 'TEXT';

                    // Generate label from resource ID or property name
                    const resourceAttr = el.getAttribute('resource') || '';
                    const label = resourceAttr.replace(/^#/, '') ||
                                getPropertyName(el) ||
                                `value_${pageIdx}_${Math.random().toString(36).substr(2, 9)}`;

                    // Capture raw RDFa attributes for graph visualization
                    const rdfaTypeof = el.getAttribute('typeof') || el.closest('[typeof]')?.getAttribute('typeof') || null;
                    const rdfaProperty = el.getAttribute('property') || null;

                    // Primary annotation (structural for cells, semantic for values)
                    const rect = isCellValue ? cellRect : elRect;
                    const annotations = [{
                        label: label,
                        entity_type: entityType,
                        text_content: el.textContent.trim(),
                        x: rect.left - pageRect.left,
                        y: rect.top - pageRect.top,
                        width: rect.width,
                        height: rect.height,
                        field_for: null,
                        field_id: effectiveFieldId,
                        section_id: effectiveSectionId,
                        table_id: tableId,
                        row_id: rowId,
                        column_header: columnHeader,
                        rdfa_typeof: rdfaTypeof,
                        rdfa_property: rdfaProperty
                    }];

                    // Dual annotation: cellValue elements with a semantic type
                    // also emit a NER annotation so entity types in table cells are captured
                    if (isCellValue && structuralType) {{
                        const semanticType = resolveSemanticType(el);
                        if (semanticType) {{
                            annotations.push({
                                label: label,
                                entity_type: semanticType,
                                text_content: el.textContent.trim(),
                                x: elRect.left - pageRect.left,
                                y: elRect.top - pageRect.top,
                                width: elRect.width,
                                height: elRect.height,
                                field_for: null,
                                field_id: effectiveFieldId,
                                section_id: effectiveSectionId,
                                table_id: tableId,
                                row_id: rowId,
                                column_header: columnHeader,
                                rdfa_typeof: rdfaTypeof,
                                rdfa_property: rdfaProperty
                            });
                        }}
                    }}

                    return annotations;
                });

                return {
                    page_number: pageIdx + 1,
                    width: pageRect.width,
                    height: pageRect.height,
                    sections: sections,
                    fields: fields,
                    annotations: boxes
                };
            });
        }"""
        
        page_data = page.evaluate(js_code)

        return page_data
    
    def screenshot_pages(
        self,
        page: Page,
        output_dir: Path,
        base_name: str,
        image_format: str = 'jpg',
        quality: int = 95
    ) -> List[str]:
        """
        Take screenshots of each paginated page.

        Args:
            page: Playwright page instance (after pagination)
            output_dir: Directory to save screenshots
            base_name: Base filename (e.g., "invoice_001")
            image_format: 'jpeg' or 'png'
            quality: Image quality (1-100 for JPEG)

        Returns:
            List of screenshot file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Map common extensions to Playwright's expected format names
        format_map = {
            'jpg': 'jpeg',
            'jpeg': 'jpeg',
            'png': 'png'
        }
        playwright_format = format_map.get(image_format.lower(), 'jpeg')

        page_elements = page.query_selector_all('.pagedjs_page')
        screenshot_paths = []

        for idx, page_el in enumerate(page_elements):
            page_num = idx + 1

            # Generate filename
            if len(page_elements) == 1:
                filename = f"{base_name}.{image_format}"
            else:
                filename = f"{base_name}_page_{page_num}.{image_format}"

            filepath = output_dir / filename

            # Take screenshot
            page_el.screenshot(
                path=str(filepath),
                type=playwright_format,
                quality=quality if playwright_format == 'jpeg' else None
            )
            
            screenshot_paths.append(str(filepath))
        
        return screenshot_paths
    
    def save_as_pdf(
        self,
        screenshot_paths: List[str],
        output_path: Path,
        dpi: int = 150
    ) -> str:
        """
        Save screenshots as a multi-page PDF with proper margins at higher DPI.

        Args:
            screenshot_paths: List of screenshot file paths to combine into PDF
            output_path: Path to save PDF
            dpi: DPI for PDF output (default 150 for better quality)

        Returns:
            Path to saved PDF
        """
        from PIL import Image

        if not screenshot_paths:
            raise ValueError("No screenshots provided for PDF generation")

        # PagedJS already handles page layout and margins via CSS
        # Just use screenshots as-is without adding extra margins
        pdf_pages = []
        for img_path in screenshot_paths:
            # Load screenshot
            screenshot = Image.open(img_path)

            # Convert to RGB if needed
            if screenshot.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', screenshot.size, (255, 255, 255))
                if screenshot.mode == 'P':
                    screenshot = screenshot.convert('RGBA')
                rgb_img.paste(screenshot, mask=screenshot.split()[-1] if screenshot.mode in ('RGBA', 'LA') else None)
                screenshot = rgb_img
            else:
                screenshot = screenshot.convert('RGB')

            pdf_pages.append(screenshot)

        # Save as PDF at specified DPI
        if len(pdf_pages) == 1:
            pdf_pages[0].save(str(output_path), 'PDF', resolution=float(dpi))
        else:
            pdf_pages[0].save(
                str(output_path),
                'PDF',
                resolution=float(dpi),
                save_all=True,
                append_images=pdf_pages[1:]
            )

        # Close all pages
        for page in pdf_pages:
            page.close()

        return str(output_path)

    def pdf_to_images(
        self,
        pdf_path: Path,
        output_dir: Path,
        base_name: str,
        dpi: int = 150,
        image_format: str = 'jpg',
        quality: int = 95
    ) -> List[str]:
        """
        Convert PDF pages to images using PyMuPDF.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            base_name: Base filename for images
            dpi: DPI for image conversion
            image_format: 'jpeg' or 'png'
            quality: JPEG quality (1-100)

        Returns:
            List of image file paths
        """
        import fitz  # PyMuPDF
        from PIL import Image
        import io

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open PDF
        pdf_doc = fitz.open(str(pdf_path))
        page_count = pdf_doc.page_count

        # Calculate zoom factor from DPI
        # PyMuPDF uses 72 DPI as base
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        image_paths = []
        for idx in range(page_count):
            page_num = idx + 1
            page = pdf_doc[idx]

            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))

            # Generate filename
            if page_count == 1:
                filename = f"{base_name}.{image_format}"
            else:
                filename = f"{base_name}_page_{page_num}.{image_format}"

            filepath = output_dir / filename

            # Save image
            if image_format == 'jpeg':
                img.save(str(filepath), 'JPEG', quality=quality)
            else:
                img.save(str(filepath), 'PNG')

            image_paths.append(str(filepath))

        pdf_doc.close()
        return image_paths

    def create_overlay_images(
        self,
        image_paths: List[str],
        page_data: List[Dict[str, Any]],
        output_dir: Path,
        base_name: str,
        margin_pixels: int,
        image_format: str = 'jpg',
        quality: int = 95
    ) -> List[str]:
        """
        Create overlay images with bounding boxes drawn on them.

        Args:
            image_paths: List of image file paths
            page_data: List of page data with annotations
            output_dir: Directory to save overlay images
            base_name: Base filename for overlay images
            margin_pixels: Margin size in pixels (to adjust coordinates)
            image_format: 'jpeg' or 'png'
            quality: JPEG quality (1-100)

        Returns:
            List of overlay image file paths
        """
        from PIL import Image, ImageDraw, ImageFont

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try to load a decent font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", 16)
            except:
                font = ImageFont.load_default()

        overlay_paths = []

        for idx, (img_path, page_info) in enumerate(zip(image_paths, page_data)):
            # Load image
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img, 'RGBA')

            # Get content area dimensions from page_info
            content_width = page_info.get('width', 816)
            content_height = page_info.get('height', 1056)

            # Calculate actual content area in final image (image size - margins)
            actual_content_width = img.size[0] - (2 * margin_pixels)
            actual_content_height = img.size[1] - (2 * margin_pixels)

            # Draw each annotation
            for ann in page_info.get('annotations', []):
                box = ann.get('box_2d', [])
                if len(box) == 4:
                    # box_2d is in normalized 0-1000 scale
                    # Convert to actual content pixels, then scale to final image size, then add margin

                    # Step 1: Convert from 0-1000 normalized to original content coordinates
                    x1_norm = (box[0] / 1000.0) * content_width
                    y1_norm = (box[1] / 1000.0) * content_height
                    x2_norm = (box[2] / 1000.0) * content_width
                    y2_norm = (box[3] / 1000.0) * content_height

                    # Step 2: Scale to actual image content area
                    scale_x = actual_content_width / content_width
                    scale_y = actual_content_height / content_height

                    x1_scaled = x1_norm * scale_x
                    y1_scaled = y1_norm * scale_y
                    x2_scaled = x2_norm * scale_x
                    y2_scaled = y2_norm * scale_y

                    # Step 3: Add margin offset
                    x1 = int(x1_scaled + margin_pixels)
                    y1 = int(y1_scaled + margin_pixels)
                    x2 = int(x2_scaled + margin_pixels)
                    y2 = int(y2_scaled + margin_pixels)

                    # Draw semi-transparent bounding box
                    # Draw filled rectangle with alpha
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3, fill=(255, 0, 0, 30))

                    # Draw label with background
                    label = ann.get('label', 'unknown')
                    entity_type = ann.get('entity_type', '')

                    # Strip numeric suffixes from label (e.g., "merchant_1" -> "merchant")
                    import re
                    clean_label = re.sub(r'_\d+$', '', label)

                    # Use entity type as primary label if available
                    if entity_type:
                        label_text = f"{entity_type}"
                    else:
                        label_text = clean_label

                    # Calculate text size
                    bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Position label in top-right corner of bbox
                    label_x = x2 - text_width - 6
                    label_y = y1 + 2

                    # Draw background rectangle for label
                    draw.rectangle(
                        [label_x - 3, label_y - 2, label_x + text_width + 3, label_y + text_height + 2],
                        fill=(255, 255, 255, 230),
                        outline=(255, 0, 0, 255),
                        width=1
                    )

                    # Draw label text
                    draw.text((label_x, label_y), label_text, fill=(0, 0, 0, 255), font=font)

            # Save overlay image
            page_num = idx + 1
            if len(image_paths) == 1:
                filename = f"{base_name}_overlay.{image_format}"
            else:
                filename = f"{base_name}_page_{page_num}_overlay.{image_format}"

            filepath = output_dir / filename

            # Convert back to RGB for JPEG
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                img = rgb_img

            if image_format == 'jpeg':
                img.save(str(filepath), 'JPEG', quality=quality)
            else:
                img.save(str(filepath), 'PNG')

            overlay_paths.append(str(filepath))

        return overlay_paths

    def create_test_metrics(
        self,
        pdf_path: Path,
        page_data: List[Dict[str, Any]],
        html_content: str,
        output_dir: Path,
        base_name: str,
        margin_pixels: int,
        dpi: int = 150
    ) -> Dict[str, str]:
        """
        Create comprehensive test metrics with detailed visual guides and diagnostics.

        Args:
            pdf_path: Path to source PDF
            page_data: List of page data with annotations
            html_content: Original HTML content
            output_dir: Directory for test outputs
            base_name: Base name for files
            margin_pixels: Margin size in pixels
            dpi: DPI for rendering

        Returns:
            Dictionary with paths to test outputs
        """
        import fitz
        from PIL import Image, ImageDraw, ImageFont
        import io

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load fonts
        try:
            font_large = ImageFont.truetype("arial.ttf", 16)
            font_medium = ImageFont.truetype("arial.ttf", 14)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                font_large = ImageFont.truetype("Arial.ttf", 16)
                font_medium = ImageFont.truetype("Arial.ttf", 14)
                font_small = ImageFont.truetype("Arial.ttf", 12)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()

        # Save HTML
        html_path = output_dir / f"{base_name}_test.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Open PDF
        pdf_doc = fitz.open(str(pdf_path))
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        test_images = []

        # Collect overall statistics
        total_annotations = sum(len(p.get('annotations', [])) for p in page_data)
        entity_type_counts = {}
        for page_info in page_data:
            for ann in page_info.get('annotations', []):
                entity_type = ann.get('entity_type') or 'unknown'
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        # Process ALL pages for test output
        num_test_pages = len(page_data)

        for idx in range(num_test_pages):
            page = pdf_doc[idx]
            page_info = page_data[idx]

            # Render page
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            draw = ImageDraw.Draw(img)

            # Get dimensions
            img_width, img_height = img.size
            content_width = page_info.get('width', 816)
            content_height = page_info.get('height', 1056)
            actual_content_width = img_width - (2 * margin_pixels)
            actual_content_height = img_height - (2 * margin_pixels)

            # Calculate scale factors for coordinate transformation
            scale_x = actual_content_width / content_width
            scale_y = actual_content_height / content_height

            # Draw margin guides (blue)
            draw.rectangle([margin_pixels, margin_pixels,
                          img_width - margin_pixels - 1,
                          img_height - margin_pixels - 1],
                         outline='blue', width=5)

            # Draw corner crosshairs (green)
            cross_len = 50
            # Top-left
            draw.line([margin_pixels, 0, margin_pixels, cross_len], fill='green', width=3)
            draw.line([0, margin_pixels, cross_len, margin_pixels], fill='green', width=3)
            # Top-right
            draw.line([img_width - margin_pixels - 1, 0, img_width - margin_pixels - 1, cross_len], fill='green', width=3)
            draw.line([img_width - cross_len, margin_pixels, img_width, margin_pixels], fill='green', width=3)
            # Bottom-left
            draw.line([margin_pixels, img_height - cross_len, margin_pixels, img_height], fill='green', width=3)
            draw.line([0, img_height - margin_pixels - 1, cross_len, img_height - margin_pixels - 1], fill='green', width=3)
            # Bottom-right
            draw.line([img_width - margin_pixels - 1, img_height - cross_len,
                      img_width - margin_pixels - 1, img_height], fill='green', width=3)
            draw.line([img_width - cross_len, img_height - margin_pixels - 1,
                      img_width, img_height - margin_pixels - 1], fill='green', width=3)

            # Draw ALL bounding boxes with labels
            annotations = page_info.get('annotations', [])

            for ann_idx, ann in enumerate(annotations):
                box = ann.get('box_2d', [])
                if len(box) == 4:
                    # Convert coordinates
                    x1_norm = (box[0] / 1000.0) * content_width
                    y1_norm = (box[1] / 1000.0) * content_height
                    x2_norm = (box[2] / 1000.0) * content_width
                    y2_norm = (box[3] / 1000.0) * content_height

                    x1 = int(x1_norm * scale_x + margin_pixels)
                    y1 = int(y1_norm * scale_y + margin_pixels)
                    x2 = int(x2_norm * scale_x + margin_pixels)
                    y2 = int(y2_norm * scale_y + margin_pixels)

                    # Draw semi-transparent box in red (matching overlay style)
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=2)

                    # Draw label with background (matching overlay style)
                    label = ann.get('label', 'unknown')
                    entity_type = ann.get('entity_type', '')

                    # Strip numeric suffixes from label (e.g., "merchant_1" -> "merchant")
                    import re
                    clean_label = re.sub(r'_\d+$', '', label)

                    # Use entity type as primary label if available
                    if entity_type:
                        label_text = f"{entity_type}"
                    else:
                        label_text = clean_label

                    # Calculate label position (top-right corner)
                    try:
                        bbox = draw.textbbox((0, 0), label_text, font=font_small)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    except:
                        text_width = len(label_text) * 7
                        text_height = 12

                    label_x = x2 - text_width - 6
                    label_y = y1 + 2

                    # Ensure label stays on page
                    if label_x < margin_pixels:
                        label_x = x1 + 2
                    if label_y < margin_pixels:
                        label_y = y2 - text_height - 4

                    # Draw background rectangle for label
                    draw.rectangle(
                        [label_x - 3, label_y - 2, label_x + text_width + 3, label_y + text_height + 2],
                        fill=(255, 255, 255, 230),
                        outline=(255, 0, 0, 255),
                        width=1
                    )

                    # Draw label text
                    draw.text((label_x, label_y), label_text, fill=(0, 0, 0, 255), font=font_small)

            # Add comprehensive metrics text overlay with background
            metrics_x = 10
            metrics_y = 10

            # Build metrics lines
            metrics_text = [
                f"PAGE {idx + 1} OF {num_test_pages} - TEST METRICS",
                f"",
                f"Image Dimensions:",
                f"  Size: {img_width}x{img_height}px @ {dpi} DPI",
                f"  Margin: {margin_pixels}px ({margin_pixels/dpi:.2f}\")",
                f"  Content Area: {actual_content_width}x{actual_content_height}px",
                f"  Original Content: {content_width}x{content_height}px",
                f"  Scale Factors: {scale_x:.3f}x, {scale_y:.3f}y",
                f"",
                f"Annotations on This Page:",
                f"  Total: {len(annotations)} bounding boxes",
            ]

            # Add entity type breakdown for this page
            page_entity_counts = {}
            for ann in annotations:
                entity_type = ann.get('entity_type') or 'unknown'
                page_entity_counts[entity_type] = page_entity_counts.get(entity_type, 0) + 1

            if page_entity_counts:
                metrics_text.append("  Entity Types:")
                for entity_type, count in sorted(page_entity_counts.items()):
                    metrics_text.append(f"    {entity_type}: {count}")

            # Add document-wide statistics
            metrics_text.extend([
                f"",
                f"Document-Wide Statistics:",
                f"  Total Pages: {num_test_pages}",
                f"  Total Annotations: {total_annotations}",
                f"  Avg per Page: {total_annotations/num_test_pages:.1f}",
            ])

            if entity_type_counts:
                metrics_text.append("  All Entity Types:")
                for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
                    metrics_text.append(f"    {entity_type}: {count}")

            # Calculate background size for metrics panel
            max_width = 0
            total_height = 0
            line_heights = []
            for line in metrics_text:
                if line:
                    try:
                        bbox = draw.textbbox((0, 0), line, font=font_small)
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                    except:
                        width = len(line) * 7
                        height = 12
                else:
                    width = 0
                    height = 8  # Blank line
                max_width = max(max_width, width)
                line_heights.append(height)
                total_height += height + 4

            # Draw semi-transparent background for metrics panel
            panel_padding = 10
            draw.rectangle(
                [metrics_x - panel_padding, metrics_y - panel_padding,
                 metrics_x + max_width + panel_padding, metrics_y + total_height + panel_padding],
                fill=(255, 255, 255, 220),
                outline=(0, 0, 0, 255),
                width=2
            )

            # Draw metrics text
            current_y = metrics_y
            for line, line_height in zip(metrics_text, line_heights):
                if line:
                    # Use bold font for headers
                    if line.endswith(':') and not line.startswith(' '):
                        draw.text((metrics_x, current_y), line, fill=(0, 0, 0, 255), font=font_medium)
                    else:
                        draw.text((metrics_x, current_y), line, fill=(0, 0, 0, 255), font=font_small)
                current_y += line_height + 4

            if img.mode != 'RGB':
                img = img.convert('RGB')

            test_images.append(img)

        pdf_doc.close()

        # Save test PDF
        test_pdf_path = output_dir / f"{base_name}_test.pdf"
        if len(test_images) == 1:
            test_images[0].save(str(test_pdf_path), 'PDF', resolution=float(dpi))
        else:
            test_images[0].save(
                str(test_pdf_path),
                'PDF',
                resolution=float(dpi),
                save_all=True,
                append_images=test_images[1:]
            )

        # Save individual test images
        test_img_paths = []
        for idx, img in enumerate(test_images):
            img_path = output_dir / f"{base_name}_test_page_{idx+1}.jpg"
            img.save(str(img_path), 'JPEG', quality=95)
            test_img_paths.append(str(img_path))
            img.close()

        return {
            'test_pdf': str(test_pdf_path),
            'test_images': test_img_paths,
            'test_html': str(html_path)
        }

    def create_pdf_overlays(
        self,
        pdf_path: Path,
        page_data: List[Dict[str, Any]],
        output_path: Path,
        margin_pixels: int,
        dpi: int = 150
    ) -> str:
        """
        Create a PDF with bounding boxes overlaid on each page.

        Args:
            pdf_path: Path to source PDF file
            page_data: List of page data with annotations
            output_path: Path to save overlay PDF
            margin_pixels: Margin size in pixels
            dpi: DPI for rendering

        Returns:
            Path to saved overlay PDF
        """
        import fitz  # PyMuPDF
        from PIL import Image, ImageDraw, ImageFont
        import io

        # Try to load a decent font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", 16)
            except:
                font = ImageFont.load_default()

        # Open source PDF
        pdf_doc = fitz.open(str(pdf_path))

        # Calculate zoom factor from DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        # Create images with overlays
        overlay_images = []

        for idx, page_info in enumerate(page_data):
            if idx >= pdf_doc.page_count:
                break

            page = pdf_doc[idx]

            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            draw = ImageDraw.Draw(img, 'RGBA')

            # Get content area dimensions
            content_width = page_info.get('width', 816)
            content_height = page_info.get('height', 1056)

            # Calculate actual content area in image
            actual_content_width = img.size[0] - (2 * margin_pixels)
            actual_content_height = img.size[1] - (2 * margin_pixels)

            # Draw each annotation
            for ann in page_info.get('annotations', []):
                box = ann.get('box_2d', [])
                if len(box) == 4:
                    # Convert coordinates (same logic as create_overlay_images)
                    x1_norm = (box[0] / 1000.0) * content_width
                    y1_norm = (box[1] / 1000.0) * content_height
                    x2_norm = (box[2] / 1000.0) * content_width
                    y2_norm = (box[3] / 1000.0) * content_height

                    scale_x = actual_content_width / content_width
                    scale_y = actual_content_height / content_height

                    x1_scaled = x1_norm * scale_x
                    y1_scaled = y1_norm * scale_y
                    x2_scaled = x2_norm * scale_x
                    y2_scaled = y2_norm * scale_y

                    x1 = int(x1_scaled + margin_pixels)
                    y1 = int(y1_scaled + margin_pixels)
                    x2 = int(x2_scaled + margin_pixels)
                    y2 = int(y2_scaled + margin_pixels)

                    # Draw semi-transparent bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3, fill=(255, 0, 0, 30))

                    # Draw label with background
                    label = ann.get('label', 'unknown')
                    entity_type = ann.get('entity_type', '')

                    # Strip numeric suffixes from label (e.g., "merchant_1" -> "merchant")
                    import re
                    clean_label = re.sub(r'_\d+$', '', label)

                    # Use entity type as primary label if available
                    if entity_type:
                        label_text = f"{entity_type}"
                    else:
                        label_text = clean_label

                    # Calculate text size
                    bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Position label in top-right corner of bbox
                    label_x = x2 - text_width - 6
                    label_y = y1 + 2

                    # Draw background rectangle for label
                    draw.rectangle(
                        [label_x - 3, label_y - 2, label_x + text_width + 3, label_y + text_height + 2],
                        fill=(255, 255, 255, 230),
                        outline=(255, 0, 0, 255),
                        width=1
                    )

                    # Draw label text
                    draw.text((label_x, label_y), label_text, fill=(0, 0, 0, 255), font=font)

            # Convert to RGB if needed
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            overlay_images.append(img)

        pdf_doc.close()

        # Save as PDF
        if len(overlay_images) == 1:
            overlay_images[0].save(str(output_path), 'PDF', resolution=float(dpi))
        else:
            overlay_images[0].save(
                str(output_path),
                'PDF',
                resolution=float(dpi),
                save_all=True,
                append_images=overlay_images[1:]
            )

        # Close images
        for img in overlay_images:
            img.close()

        return str(output_path)

    def render_and_extract(
        self,
        html_path: Path,
        output_dir: Optional[Path] = None,
        screenshot: bool = True,
        base_name: Optional[str] = None,
        save_pdf: bool = False,
        pdf_output_dir: Optional[Path] = None,
        create_overlays: bool = False,
        overlay_output_dir: Optional[Path] = None,
        create_test_output: bool = False,
        test_output_dir: Optional[Path] = None,
        dpi: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Main entry point: Render HTML with Paged.js and extract data.

        Args:
            html_path: Path to HTML file
            output_dir: Directory for screenshots/images (if screenshot=True)
            screenshot: Whether to save page images
            base_name: Base name for output files (defaults to html filename)
            save_pdf: Whether to save PDF
            pdf_output_dir: Directory for PDF (defaults to output_dir)
            create_overlays: Whether to create overlay images with bounding boxes
            overlay_output_dir: Directory for overlay images (defaults to output_dir/overlays)
            dpi: DPI for PDF and image generation (default 150)

        Returns:
            List of page data dictionaries containing:
                - page_number: Page index (1-based)
                - width: Page width in pixels (content area, no margins)
                - height: Page height in pixels (content area, no margins)
                - annotations: List of bounding box annotations
                - image_path: Path to final image (if screenshot=True)
                - pdf_path: Path to PDF (if save_pdf=True, only in first page)
                - overlay_path: Path to overlay image (if create_overlays=True)
        """
        html_path = Path(html_path)
        
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_path}")
        
        # Read HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Inject Paged.js and ensure @page CSS
        html_content = self.inject_pagedjs_script(html_content)
        html_content = self.ensure_page_css(html_content)
        
        # Save modified HTML to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.html',
            delete=False,
            encoding='utf-8'
        ) as temp_file:
            temp_file.write(html_content)
            temp_html_path = Path(temp_file.name)
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                page = browser.new_page(
                    viewport={
                        'width': self.viewport_width,
                        'height': self.viewport_height
                    }
                )
                
                # Load HTML
                page.goto(f'file:///{temp_html_path.as_posix()}', wait_until='networkidle')
                
                # Wait for Paged.js pagination
                page_count = self.wait_for_pagination(page)
                print(f"Paged.js created {page_count} pages")
                
                # Extract data
                page_data = self.extract_page_data(page)

                # Print extraction summary
                total_annotations = sum(len(p.get('annotations', [])) for p in page_data)
                if total_annotations > 0:
                    print(f"Extracted {total_annotations} annotations from RDFa markup")

                # Set default values
                if output_dir is None:
                    output_dir = html_path.parent
                if base_name is None:
                    base_name = html_path.stem

                # Take content-only screenshots (no margins). A private directory per
                # render, so concurrent runs that reuse base names cannot collide.
                screenshot_dir = Path(tempfile.mkdtemp(prefix="synthdoc_"))
                temp_screenshot_paths = self.screenshot_pages(
                    page,
                    screenshot_dir,
                    f"{base_name}_temp"
                )

                browser.close()

                # Save PDF if requested (PagedJS handles margins via CSS)
                pdf_path = None
                if save_pdf:
                    # Determine PDF output directory
                    if pdf_output_dir is not None:
                        pdf_dir = Path(pdf_output_dir)
                    elif output_dir is not None:
                        pdf_dir = Path(output_dir)
                    else:
                        pdf_dir = html_path.parent

                    pdf_dir.mkdir(parents=True, exist_ok=True)
                    pdf_path = pdf_dir / f"{base_name}.pdf"
                    self.save_as_pdf(temp_screenshot_paths, pdf_path, dpi=dpi)

                    # Add to first page data
                    if page_data:
                        page_data[0]['pdf_path'] = str(pdf_path)

                # Generate images from PDF (or use screenshots if no PDF)
                final_image_paths = []
                if screenshot:
                    img_dir = Path(output_dir)
                    img_dir.mkdir(parents=True, exist_ok=True)

                    if save_pdf and pdf_path:
                        # Convert PDF to images (includes margins)
                        final_image_paths = self.pdf_to_images(
                            pdf_path,
                            img_dir,
                            base_name,
                            dpi=dpi
                        )
                    else:
                        # Use temp screenshots (no margins)
                        for idx, temp_path in enumerate(temp_screenshot_paths):
                            page_num = idx + 1
                            if len(temp_screenshot_paths) == 1:
                                filename = f"{base_name}.jpg"
                            else:
                                filename = f"{base_name}_page_{page_num}.jpg"

                            final_path = img_dir / filename
                            shutil.copy(temp_path, final_path)
                            final_image_paths.append(str(final_path))

                    # Add image paths to page data
                    for page_info, img_path in zip(page_data, final_image_paths):
                        page_info['image_path'] = img_path

                # Create overlay images if requested
                if create_overlays and final_image_paths:
                    # Determine overlay output directory
                    if overlay_output_dir is not None:
                        overlay_dir = Path(overlay_output_dir)
                    else:
                        overlay_dir = Path(output_dir) / "overlays"

                    overlay_dir.mkdir(parents=True, exist_ok=True)

                    # Create overlays (no margin adjustment needed - PagedJS handles margins)

                    # Add normalized box_2d coordinates to annotations for overlay rendering
                    # CRITICAL: Use actual image dimensions, not browser page dimensions!
                    from PIL import Image
                    for page_info, img_path in zip(page_data, final_image_paths):
                        # Save browser dimensions before overwriting
                        browser_width = page_info.get('width', 816)
                        browser_height = page_info.get('height', 1056)

                        # Read actual image dimensions
                        with Image.open(img_path) as img:
                            actual_width, actual_height = img.size

                        # Update page dimensions to match actual image
                        page_info['width'] = actual_width
                        page_info['height'] = actual_height

                        for ann in page_info.get('annotations', []):
                            if 'box_2d' not in ann:
                                # Convert from browser coordinates to image coordinates
                                # Then normalize to 0-1000 scale
                                scale_x = actual_width / browser_width
                                scale_y = actual_height / browser_height

                                image_x = ann['x'] * scale_x
                                image_y = ann['y'] * scale_y
                                image_x2 = (ann['x'] + ann['width']) * scale_x
                                image_y2 = (ann['y'] + ann['height']) * scale_y

                                ann['box_2d'] = [
                                    int((image_x / actual_width) * 1000),
                                    int((image_y / actual_height) * 1000),
                                    int((image_x2 / actual_width) * 1000),
                                    int((image_y2 / actual_height) * 1000)
                                ]

                    overlay_paths = self.create_overlay_images(
                        final_image_paths,
                        page_data,
                        overlay_dir,
                        base_name,
                        0  # No margin adjustment - PagedJS handles margins
                    )

                    # Add overlay paths to page data
                    for page_info, overlay_path in zip(page_data, overlay_paths):
                        page_info['overlay_path'] = overlay_path

                    # Create PDF overlay if we have a PDF
                    if save_pdf and pdf_path:
                        pdf_overlay_path = overlay_dir / f"{base_name}_overlay.pdf"
                        self.create_pdf_overlays(
                            pdf_path,
                            page_data,
                            pdf_overlay_path,
                            0,  # No margins - PagedJS handles margins
                            dpi
                        )
                        # Add to first page data
                        if page_data:
                            page_data[0]['pdf_overlay_path'] = str(pdf_overlay_path)

                # Create test outputs if requested
                if create_test_output and save_pdf and pdf_path:
                    # Determine test output directory
                    if test_output_dir is not None:
                        test_dir = Path(test_output_dir)
                    else:
                        test_dir = Path(output_dir).parent / "test_output"

                    test_dir.mkdir(parents=True, exist_ok=True)

                    # Read original HTML (before Paged.js injection)
                    with open(html_path, 'r', encoding='utf-8') as f:
                        original_html = f.read()

                    # PagedJS handles margins internally
                    margin_pixels = 0

                    test_metrics = self.create_test_metrics(
                        pdf_path,
                        page_data,
                        original_html,
                        test_dir,
                        base_name,
                        margin_pixels,
                        dpi
                    )

                    # Add to first page data
                    if page_data:
                        page_data[0]['test_metrics'] = test_metrics

                # Clean up temp screenshots
                shutil.rmtree(screenshot_dir, ignore_errors=True)
            
            return page_data
        
        finally:
            # Clean up temp file
            temp_html_path.unlink()
    
    def render_to_json(
        self,
        html_path: Path,
        output_dir: Path,
        base_name: Optional[str] = None,
        normalize_coords: bool = True
    ) -> List[str]:
        """
        Render HTML and save annotations as JSON files (one per page).
        
        Args:
            html_path: Path to HTML file
            output_dir: Directory for output files
            base_name: Base name for output files
            normalize_coords: Normalize coordinates to 0-1000 scale
            
        Returns:
            List of JSON file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if base_name is None:
            base_name = Path(html_path).stem
        
        # Render and extract
        page_data = self.render_and_extract(html_path, output_dir, screenshot=True, base_name=base_name)
        
        json_paths = []
        
        for page_info in page_data:
            page_num = page_info['page_number']
            
            # Normalize coordinates if requested
            if normalize_coords:
                for ann in page_info['annotations']:
                    page_width = page_info['width']
                    page_height = page_info['height']
                    
                    # Convert to 0-1000 scale
                    ann['box_2d'] = [
                        int((ann['x'] / page_width) * 1000),
                        int((ann['y'] / page_height) * 1000),
                        int((ann['x'] + ann['width']) / page_width * 1000),
                        int((ann['y'] + ann['height']) / page_height * 1000)
                    ]
                    
                    # Remove raw coordinates
                    del ann['x']
                    del ann['y']
                    del ann['width']
                    del ann['height']
            
            # Create JSON filename
            if len(page_data) == 1:
                json_filename = f"{base_name}.json"
            else:
                json_filename = f"{base_name}_page_{page_num}.json"
            
            json_path = output_dir / json_filename
            
            # Save JSON
            output_data = {
                'page_number': page_num,
                'width': int(page_info['width']),
                'height': int(page_info['height']),
                'image_path': page_info.get('image_path'),
                'annotations': page_info['annotations']
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            json_paths.append(str(json_path))
        
        return json_paths


# Convenience functions for backward compatibility

def render_html_with_pagedjs(
    html_path: Path,
    output_dir: Path,
    base_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function: Render HTML with Paged.js (similar to old API).
    
    Args:
        html_path: Path to HTML file
        output_dir: Output directory
        base_name: Base name for outputs
        
    Returns:
        List of page data dictionaries
    """
    renderer = PagedJSRenderer()
    return renderer.render_to_json(html_path, output_dir, base_name)


if __name__ == "__main__":
    # Example usage
    print("Paged.js Renderer Library")
    print("=" * 60)
    print("\nUsage example:")
    print("""
from synthdoc.pagedjs_renderer import PagedJSRenderer

# Initialize renderer
renderer = PagedJSRenderer()

# Render HTML and extract data
pages = renderer.render_and_extract(
    html_path='document.html',
    output_dir='output/',
    base_name='my_document'
)

# Access data
for page in pages:
    print(f"Page {page['page_number']}: {len(page['annotations'])} boxes")
    for ann in page['annotations']:
        print(f"  {ann['label']}: {ann['text_content'][:30]}...")
""")
