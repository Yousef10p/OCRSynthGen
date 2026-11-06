"""
NER Data Generator for creating training datasets with pixel-perfect ground truth
using HTML/CSS templates and Paged.js for pagination.
"""
import json
from pathlib import Path
from typing import Dict, List, Any
import tempfile
from dataclasses import dataclass

from jinja2 import Template
from faker import Faker


@dataclass
class NERBox:
    """Data class for NER bounding boxes"""
    label: str
    text: str
    x: float
    y: float
    width: float
    height: float


class NERDataGenerator:
    """Generates NER training data with pixel-perfect ground truth coordinates using Paged.js"""

    # Class-level cache for entity hierarchy
    _entity_hierarchy = None

    def __init__(self, output_dir: str = "ner_dataset", create_overlays: bool = True,
                 create_test_output: bool = False, entity_types: List[str] = None,
                 min_bbox_size: int = 5, bbox_padding: int = 0,
                 filter_tiny_boxes: bool = True, output_formats: List[str] = None,
                 images_subdir: str = "images", labels_subdir: str = "labels",
                 pdfs_subdir: str = "pdfs", overlays_subdir: str = "overlays",
                 html_subdir: str = "html", dpi: int = 96):
        """
        Initialize NER Data Generator.

        Args:
            output_dir: Directory for output images and labels
            create_overlays: Whether to create overlay images with bounding boxes
            create_test_output: Whether to create test output with metrics and diagnostics
            entity_types: List of entity types to include (None = all)
            min_bbox_size: Minimum bbox size in normalized units (default 5)
            bbox_padding: Padding around bboxes in pixels (default 0)
            filter_tiny_boxes: Whether to filter out boxes smaller than min_bbox_size
            output_formats: List of output formats ('json', 'ttl', 'jsonld'). Default: all
            images_subdir: Subdirectory name for images (default: 'images')
            labels_subdir: Subdirectory name for labels (default: 'labels')
            pdfs_subdir: Subdirectory name for PDFs (default: 'pdfs')
            overlays_subdir: Subdirectory name for overlays (default: 'overlays')
            html_subdir: Subdirectory name for HTML files (default: 'html')
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / images_subdir
        self.labels_dir = self.output_dir / labels_subdir
        self.pdfs_dir = self.output_dir / pdfs_subdir
        self.overlays_dir = self.output_dir / overlays_subdir
        self.html_dir = self.output_dir / html_subdir
        self.test_output_dir = self.output_dir / "test_output"
        self.create_overlays = create_overlays
        self.create_test_output = create_test_output
        self.entity_types = entity_types
        self.min_bbox_size = min_bbox_size
        self.bbox_padding = bbox_padding
        self.filter_tiny_boxes = filter_tiny_boxes
        self.output_formats = output_formats if output_formats else ['json', 'ttl', 'jsonld']
        self.dpi = dpi

        # Create output directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        if create_overlays:
            self.overlays_dir.mkdir(parents=True, exist_ok=True)
        if create_test_output:
            self.test_output_dir.mkdir(parents=True, exist_ok=True)

        self.fake = Faker()

        # Initialize Paged.js renderer
        from .pagedjs_renderer import PagedJSRenderer
        self.pagedjs_renderer = PagedJSRenderer(headless=True)

        # Load entity hierarchy on first initialization
        if NERDataGenerator._entity_hierarchy is None:
            NERDataGenerator._load_entity_hierarchy()

    @classmethod
    def _load_entity_hierarchy(cls):
        """Load entity type hierarchy JSON (once per class)"""
        hierarchy_file = Path(__file__).parent / "entity_type_hierarchy.json"
        if hierarchy_file.exists():
            with open(hierarchy_file, 'r') as f:
                data = json.load(f)
                cls._entity_hierarchy = data.get("entity_type_mappings", {})
        else:
            print(f"Warning: Entity hierarchy file not found at {hierarchy_file}")
            cls._entity_hierarchy = {}

    @classmethod
    def get_generic_type(cls, entity_type: str) -> str:
        """
        Get generic parent type for specific entity type.

        Args:
            entity_type: Specific entity type (e.g., "PLAINTIFF_NAME")

        Returns:
            Generic type (e.g., "PERSON_NAME") or "TEXT" if not found
        """
        if cls._entity_hierarchy is None:
            cls._load_entity_hierarchy()

        if entity_type in cls._entity_hierarchy:
            return cls._entity_hierarchy[entity_type].get("generic_type", "TEXT")
        return "TEXT"

    def generate_dataset_sample(self, template_html: str, template_data: Dict[str, Any],
                               output_id: str, keep_pdf: bool = False,
                               dpi: int = 300,
                               template_instance=None) -> Dict[str, Any]:
        """
        Generate a single dataset sample with image and ground truth coordinates.
        Uses Paged.js for automatic pagination and coordinate extraction.

        Args:
            template_html: HTML template string with data-label attributes
            template_data: Data to render into the template
            output_id: Unique identifier for this sample
            keep_pdf: Whether to keep the PDF file
            dpi: Resolution for generated images (ignored, kept for API compatibility)
            template_instance: Optional BaseTemplate instance for ontology metadata.
                             When provided and 'ontology' in output_formats,
                             emits a .ontology.json file alongside the label.

        Returns:
            Dictionary containing:
            - For single-page: image_path, label_path, annotations, dimensions, total_pages
            - For multi-page: pages (list of per-page results), total_pages, pdf_path (optional)
        """
        # Render HTML template
        t = Template(template_html)
        rendered_html = t.render(**template_data)

        # Save HTML to temp file
        temp_html = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
        temp_html.write(rendered_html)
        temp_html.close()

        try:
            # Use Paged.js renderer to handle everything
            page_data = self.pagedjs_renderer.render_and_extract(
                html_path=Path(temp_html.name),
                output_dir=self.images_dir,
                screenshot=True,
                base_name=output_id,
                save_pdf=keep_pdf,
                pdf_output_dir=self.pdfs_dir if keep_pdf else None,
                create_overlays=self.create_overlays,
                overlay_output_dir=self.overlays_dir if self.create_overlays else None,
                create_test_output=self.create_test_output,
                test_output_dir=self.test_output_dir if self.create_test_output else None,
                dpi=self.dpi
            )

            # Convert to our expected format
            results = []
            for page_info in page_data:
                page_num = page_info['page_number']

                # Add entity types to annotations and filter
                filtered_annotations = []
                annotations_before_filter = len(page_info['annotations'])
                for ann in page_info['annotations']:
                    # Ensure entity_type exists (may be missing in RDFa extraction)
                    if 'entity_type' not in ann or not ann['entity_type']:
                        ann['entity_type'] = 'TEXT'  # Default entity type
                        ann['generic_type'] = "TEXT"
                    else:
                        ann['generic_type'] = self.get_generic_type(ann['entity_type'])

                    # Filter by entity type if specified
                    if self.entity_types is not None:
                        if ann['generic_type'] not in self.entity_types:
                            continue

                    # Normalize coordinates to 0-1000 if not already
                    if 'box_2d' not in ann:
                        page_width = page_info['width']
                        page_height = page_info['height']
                        ann['box_2d'] = [
                            int((ann['x'] / page_width) * 1000),
                            int((ann['y'] / page_height) * 1000),
                            int((ann['x'] + ann['width']) / page_width * 1000),
                            int((ann['y'] + ann['height']) / page_height * 1000)
                        ]

                    # Apply bbox padding
                    if self.bbox_padding > 0:
                        ann['box_2d'][0] = max(0, ann['box_2d'][0] - self.bbox_padding)
                        ann['box_2d'][1] = max(0, ann['box_2d'][1] - self.bbox_padding)
                        ann['box_2d'][2] = min(1000, ann['box_2d'][2] + self.bbox_padding)
                        ann['box_2d'][3] = min(1000, ann['box_2d'][3] + self.bbox_padding)

                    # Filter tiny boxes if enabled
                    if self.filter_tiny_boxes:
                        width = ann['box_2d'][2] - ann['box_2d'][0]
                        height = ann['box_2d'][3] - ann['box_2d'][1]
                        if width < self.min_bbox_size or height < self.min_bbox_size:
                            continue

                    filtered_annotations.append(ann)

                # Debug logging
                print(f"  Page {page_num}: {annotations_before_filter} annotations -> {len(filtered_annotations)} after filtering")

                # Update page_info with filtered annotations
                page_info['annotations'] = filtered_annotations

                # Normalize section coordinates to 0-1000 scale
                normalized_sections = []
                page_width = page_info['width']
                page_height = page_info['height']

                for section in page_info.get('sections', []):
                    normalized_sections.append({
                        'section_id': section['section_id'],
                        'section_type': section['section_type'],
                        'box_2d': [
                            int((section['x'] / page_width) * 1000),
                            int((section['y'] / page_height) * 1000),
                            int((section['x'] + section['width']) / page_width * 1000),
                            int((section['y'] + section['height']) / page_height * 1000)
                        ]
                    })

                # Normalize field coordinates to 0-1000 scale
                normalized_fields = []
                for field in page_info.get('fields', []):
                    normalized_fields.append({
                        'field_id': field['field_id'],
                        'field_type': field['field_type'],
                        'section_id': field.get('section_id'),
                        'box_2d': [
                            int((field['x'] / page_width) * 1000),
                            int((field['y'] / page_height) * 1000),
                            int((field['x'] + field['width']) / page_width * 1000),
                            int((field['y'] + field['height']) / page_height * 1000)
                        ]
                    })

                # Create label file (JSON format)
                label_data = {
                    'document_id': f"{output_id}_page_{page_num}" if len(page_data) > 1 else output_id,
                    'page_number': page_num,
                    'total_pages': len(page_data),
                    'width': int(page_info['width']),
                    'height': int(page_info['height']),
                    'sections': normalized_sections,
                    'fields': normalized_fields,
                    'annotations': page_info['annotations']
                }

                label_filename = f"{output_id}_page_{page_num}.json" if len(page_data) > 1 else f"{output_id}.json"
                label_path = self.labels_dir / label_filename

                # Save JSON output if enabled
                if 'json' in self.output_formats:
                    with open(label_path, 'w', encoding='utf-8') as f:
                        json.dump(label_data, f, indent=2, ensure_ascii=False)

                # Save ontology-compliant JSON if enabled
                if 'ontology' in self.output_formats and template_instance is not None:
                    try:
                        from .ontology_adapter import build_ontology_document
                        ontology_doc = build_ontology_document(
                            label_data, template_instance, page_info
                        )
                        ontology_filename = label_filename.replace('.json', '.ontology.json')
                        ontology_path = self.labels_dir / ontology_filename
                        with open(ontology_path, 'w', encoding='utf-8') as f:
                            json.dump(ontology_doc, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"Warning: Failed to generate ontology JSON: {e}")

                # Create RDF graph (Turtle/JSON-LD formats) if enabled
                if 'ttl' in self.output_formats or 'jsonld' in self.output_formats:
                    try:
                        from .rdf_graph_generator import DocumentGraphGenerator

                        graph_gen = DocumentGraphGenerator()
                        document_id = f"{output_id}_page_{page_num}" if len(page_data) > 1 else output_id
                        document_type = "document"  # TODO: Extract from template metadata

                        graph_gen.create_document_graph(
                            document_id=document_id,
                            document_type=document_type,
                            page_data=page_info,
                            page_number=page_num
                        )

                        # Save RDF graph in Turtle format if enabled
                        if 'ttl' in self.output_formats:
                            rdf_filename = f"{output_id}_page_{page_num}.ttl" if len(page_data) > 1 else f"{output_id}.ttl"
                            rdf_path = self.labels_dir / rdf_filename
                            graph_gen.save(str(rdf_path), format='turtle')

                        # Save as JSON-LD if enabled
                        if 'jsonld' in self.output_formats:
                            jsonld_filename = f"{output_id}_page_{page_num}.jsonld" if len(page_data) > 1 else f"{output_id}.jsonld"
                            jsonld_path = self.labels_dir / jsonld_filename
                            graph_gen.save(str(jsonld_path), format='json-ld')

                    except Exception as e:
                        print(f"Warning: Failed to generate RDF graph: {e}")

                results.append({
                    'page_number': page_num,
                    'image_path': page_info['image_path'],
                    'label_path': str(label_path),
                    'annotations': page_info['annotations'],
                    'width': int(page_info['width']),
                    'height': int(page_info['height'])
                })

            # Return format matching old API
            if len(results) == 1:
                result = {
                    **results[0],
                    'total_pages': 1
                }
                # Add PDF path if available
                if 'pdf_path' in page_data[0]:
                    result['pdf_path'] = page_data[0]['pdf_path']
                return result
            else:
                result = {
                    'pages': results,
                    'total_pages': len(results)
                }
                # Add PDF path if available
                if page_data and 'pdf_path' in page_data[0]:
                    result['pdf_path'] = page_data[0]['pdf_path']
                return result

        finally:
            # Clean up temp file
            Path(temp_html.name).unlink()

    def generate_multiple_samples(self, template_html: str,
                                 generate_data_func,
                                 count: int = 100,
                                 keep_pdf: bool = False) -> List[Dict[str, Any]]:
        """
        Generate multiple samples for a dataset.

        Args:
            template_html: HTML template string with data-label attributes
            generate_data_func: Function that returns appropriate data for the template
            count: Number of samples to generate
            keep_pdf: Whether to keep PDF files

        Returns:
            List of results for each generated sample
        """
        results = []

        for i in range(count):
            # Generate template-specific data
            template_data = generate_data_func(i)

            result = self.generate_dataset_sample(
                template_html=template_html,
                template_data=template_data,
                output_id=f"{i:05d}",
                keep_pdf=keep_pdf
            )
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{count} samples...")

        print(f"Completed generation of {count} samples.")
        return results
