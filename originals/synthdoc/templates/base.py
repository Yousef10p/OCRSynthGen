"""
Base template classes and system for the document generator.
This provides a foundation for configurable templates with various layouts and styling options.
"""
import os
import json
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
from jinja2 import Template, Environment, FileSystemLoader
from dataclasses import dataclass, asdict
from faker import Faker


@dataclass
class TemplateConfig:
    """Configuration for template styling and layout"""
    font_family: str = "'Courier New', Courier, monospace"
    font_size: str = "12pt"  # Standard legal document font size
    width: str = "8.5in"
    height: str = "11in"
    margin: str = "0.5in"
    line_height: str = "1.5"  # 1.5x line spacing for readability
    color: str = "#000"
    background: str = "white"
    border: str = "1px solid #000"
    layout_type: str = "standard"  # standard, grid, form, narrative
    components: List[str] = None
    
    def __post_init__(self):
        if self.components is None:
            self.components = []


class BaseTemplate(ABC):
    """Abstract base template class"""

    def __init__(self, config: Optional[TemplateConfig] = None):
        self.config = config or TemplateConfig()
        self.fake = Faker()
        self.jinja_env = Environment(autoescape=True)
        self._template_dir = self._get_template_dir()

    # -- Ontology metadata (override in subclasses for accurate values) --

    def get_archetype(self) -> str:
        """Return the fundamental layout archetype for this template.

        Valid values: form, ledger, letter, invoice, report, certificate,
        card, legal, ticket, label.
        """
        return "form"

    def get_category(self) -> str:
        """Return the document category for this template.

        Valid values: financial, legal, medical, government, business,
        education, insurance, real_estate, transportation, logistics,
        manufacturing, retail, technology.
        """
        return "business"

    def get_document_type(self) -> str:
        """Return a specific document type identifier (e.g. 'invoice_commercial')."""
        return self.__class__.__name__.lower().replace("template", "")

    def _get_template_dir(self):
        """Get the directory containing template files for V2 templates."""
        import inspect
        from pathlib import Path

        # Get the module file path
        module_file = Path(inspect.getfile(self.__class__))

        # Check if we're in V2 structure (Python file inside template folder)
        parent_dir = module_file.parent
        if parent_dir.name.endswith('_template'):
            # V2 structure: we're already in the template folder
            return parent_dir

        # Old structure or V1: look for template folder next to Python file
        template_name = module_file.stem
        template_dir = parent_dir / template_name
        return template_dir if template_dir.exists() else None

    def _load_file(self, filename: str, required: bool = False) -> str:
        """Load a file from the template directory (for V2 templates)."""
        if not self._template_dir or not self._template_dir.exists():
            if required:
                raise FileNotFoundError(f"Template directory not found: {self._template_dir}")
            return ""

        file_path = self._template_dir / filename
        if not file_path.exists():
            if required:
                raise FileNotFoundError(f"Required template file not found: {file_path}")
            return ""

        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @abstractmethod
    def get_template_string(self) -> str:
        """Return the Jinja2 template string"""
        pass
        
    def render(self, data: Dict[str, Any]) -> str:
        """Render the template with provided data"""
        template_str = self.get_template_string()
        template = self.jinja_env.from_string(template_str)
        return template.render(**data, config=self.config)
        
    def get_css_styles(self) -> str:
        """Generate CSS styles based on configuration"""
        return f"""
        body {{
            font-family: {self.config.font_family};
            font-size: {self.config.font_size};
            color: {self.config.color};
            background: {self.config.background};
            margin: 0;
            padding: 0;
        }}
        .container {{
            width: {self.config.width};
            height: {self.config.height};
            margin: {self.config.margin};
            padding: {self.config.margin};
            position: relative;
        }}
        """


class Component(ABC):
    """Base component class for reusable template elements"""
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        self.config = config or TemplateConfig()
        
    @abstractmethod
    def render(self, data: Dict[str, Any]) -> str:
        """Render the component with provided data"""
        pass


class TableComponent(Component):
    """Reusable table component"""
    
    def render(self, data: Dict[str, Any]) -> str:
        """
        Render a table with dynamic data.
        
        Expected data keys:
        - headers: list of header names
        - rows: list of dictionaries with row data
        - styles: optional CSS class names for styling
        """
        headers = data.get('headers', [])
        rows = data.get('rows', [])
        styles = data.get('styles', {})
        
        # Determine column count based on headers
        col_count = len(headers)
        if col_count > 0:
            # Calculate column widths (approximately equal)
            col_width = f"{100.0 / col_count:.2f}%"
        else:
            col_width = "auto"
        
        table_html = f'<table class="{styles.get("table", "default-table")}">\n'
        
        # Render headers
        if headers:
            table_html += "  <thead>\n    <tr>\n"
            for header in headers:
                table_html += f"      <th style='width: {col_width};'>{header}</th>\n"
            table_html += "    </tr>\n  </thead>\n"
        
        # Render body
        table_html += "  <tbody>\n"
        for row in rows:
            table_html += "    <tr>\n"
            for header in headers:
                cell_value = row.get(header, "")
                table_html += f"      <td>{cell_value}</td>\n"
            table_html += "    </tr>\n"
        table_html += "  </tbody>\n</table>"
        
        return table_html


class HeaderComponent(Component):
    """Reusable header component"""
    
    def render(self, data: Dict[str, Any]) -> str:
        """
        Render a header with title and optional metadata.
        
        Expected data keys:
        - title: main title
        - metadata: optional key-value pairs to display
        """
        title = data.get('title', 'Untitled Document')
        metadata = data.get('metadata', {})
        
        header_html = f"<header>\n  <h1>{title}</h1>\n"
        
        if metadata:
            header_html += "  <div class='metadata'>\n"
            for key, value in metadata.items():
                header_html += f"    <p><strong>{key}:</strong> {value}</p>\n"
            header_html += "  </div>\n"
        
        header_html += "</header>"
        return header_html


class LineNumbersComponent(Component):
    """Specialized component for line numbers (for legal documents)"""
    
    def render(self, data: Dict[str, Any]) -> str:
        """
        Render line numbers for legal documents.
        
        Expected data keys:
        - start: starting number (default 1)
        - end: ending number (default 28)
        - position: CSS position properties (default "absolute left: 0.3in")
        """
        start = data.get('start', 1)
        end = data.get('end', 28)
        position = data.get('position', 'absolute; left: 0.3in; top: 0; bottom: 0;')
        
        line_numbers = range(start, end + 1)
        numbers_html = "<br>".join(map(str, line_numbers))
        
        return f"""<div class="line-numbers" style="{position} color: #000; font-size: {self.config.font_size}; line-height: {self.config.line_height}; text-align: right; width: 20px; border-right: 1px solid #ccc; padding-right: 5px;">
{numbers_html}
</div>"""


