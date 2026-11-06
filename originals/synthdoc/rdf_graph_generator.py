"""
RDF/OWL Graph Generator for Document Ontology

Generates RDF triples representing document structure, fields, annotations,
and their relationships without duplicating data.
"""
from typing import Dict, List, Any, Optional
from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL, XSD
from rdflib.namespace import FOAF
from urllib.parse import quote
import json


# Define namespaces
DOC = Namespace("http://example.org/document-ontology#")
BBOX = Namespace("http://example.org/bounding-box#")
INST = Namespace("http://example.org/instances/")
SCHEMA = Namespace("http://schema.org/")
DBO = Namespace("http://dbpedia.org/ontology/")
DBR = Namespace("http://dbpedia.org/resource/")


class DocumentGraphGenerator:
    """Generates RDF graphs from document extraction data"""

    def __init__(self, include_float_coords: bool = False):
        """
        Initialize RDF graph generator

        Args:
            include_float_coords: If True, include pixel float coordinates (x, y, width, height)
                                 in addition to normalized integers. Default False.
        """
        self.graph = Graph()
        self.graph.bind("doc", DOC)
        self.graph.bind("bbox", BBOX)
        self.graph.bind("inst", INST)
        self.graph.bind("owl", OWL)
        self.graph.bind("schema", SCHEMA)
        self.graph.bind("dbo", DBO)
        self.graph.bind("dbr", DBR)
        self.include_float_coords = include_float_coords

    def create_document_graph(
        self,
        document_id: str,
        document_type: str,
        page_data: List[Dict[str, Any]],
        page_number: int = 1
    ) -> Graph:
        """
        Create RDF graph from page data with document structure ontology.

        Args:
            document_id: Unique document identifier
            document_type: Type of document (e.g., 'medical_intake_form')
            page_data: Extracted page data including annotations, sections, fields
            page_number: Page number within document

        Returns:
            RDF Graph with triples representing document structure
        """
        # Create document instance
        doc_uri = INST[f"doc/{document_id}"]
        self.graph.add((doc_uri, RDF.type, DOC.Document))
        self.graph.add((doc_uri, DOC.documentId, Literal(document_id, datatype=XSD.string)))
        self.graph.add((doc_uri, DOC.documentType, Literal(document_type, datatype=XSD.string)))

        # Create page instance
        page_uri = INST[f"page/{document_id}"]
        self.graph.add((page_uri, RDF.type, DOC.Page))
        self.graph.add((page_uri, DOC.pageNumber, Literal(page_number, datatype=XSD.integer)))
        self.graph.add((doc_uri, DOC.hasPage, page_uri))

        # Process sections first
        sections = {}
        for sec in page_data.get('sections', []):
            section_id = sec.get('section_id')
            if section_id and section_id not in sections:
                sections[section_id] = self._create_section(
                    page_uri,
                    section_id,
                    sec.get('section_type', 'section')
                )

        # Process fields
        fields = {}
        for field_data in page_data.get('fields', []):
            field_id = field_data.get('field_id')
            if field_id and field_id not in fields:
                section_id = field_data.get('section_id')
                section_uri = sections.get(section_id) if section_id else page_uri
                fields[field_id] = self._create_field(
                    section_uri,
                    field_id,
                    field_data.get('field_type', 'field')
                )

        # Process text elements
        elements_by_id = {}
        for ann in page_data.get('annotations', []):
            element_id = ann.get('label', f"text_{len(elements_by_id)}")

            # Create text element (the actual thing in the document)
            element_uri = self._create_text_element(page_uri, element_id, ann)
            elements_by_id[element_id] = element_uri

        # Create field-label-value relationships
        self._create_field_relationships(elements_by_id, fields, page_data.get('annotations', []))

        return self.graph

    def _create_text_element(
        self,
        parent_uri: URIRef,
        element_id: str,
        ann_data: Dict[str, Any]
    ) -> URIRef:
        """Create a text element (the actual thing, not an annotation of it)"""
        # Clean ID - remove numeric suffixes
        import re
        clean_id = re.sub(r'_\d+$', '', element_id)
        # URL-encode the ID to handle spaces and special characters
        element_uri = INST[f"text/{quote(clean_id, safe='')}"]

        entity_type = ann_data.get('entity_type', '')
        text_content = ann_data.get('text_content', '')

        # Determine if it's a field label or value
        if entity_type == 'FIELD_LABEL':
            self.graph.add((element_uri, RDF.type, DOC.FieldLabel))
        else:
            self.graph.add((element_uri, RDF.type, DOC.FieldValue))

        # Always add base TextElement type
        self.graph.add((element_uri, RDF.type, DOC.TextElement))

        # Add semantic entity type (Person, Organization, etc.)
        if entity_type and entity_type != 'FIELD_LABEL':
            entity_class = self._get_entity_class(entity_type)
            if entity_class:
                self.graph.add((element_uri, RDF.type, entity_class))
                # Add linked data types
                self._add_linked_data_types(element_uri, entity_type)

        # Link to parent
        self.graph.add((parent_uri, DOC.containsElement, element_uri))

        # Add properties
        if text_content:
            self.graph.add((element_uri, DOC.text, Literal(text_content, datatype=XSD.string)))

        self.graph.add((element_uri, DOC.elementId, Literal(clean_id, datatype=XSD.string)))

        # Attach bounding box coordinates to element
        if 'box_2d' in ann_data or ('x' in ann_data and 'y' in ann_data):
            bbox_uri = self._create_bounding_box(clean_id, ann_data)
            self.graph.add((element_uri, DOC.hasBoundingBox, bbox_uri))

        return element_uri


    def _create_section(self, parent_uri: URIRef, section_id: str, section_type: str) -> URIRef:
        """Create section instance"""
        # URL-encode the ID to handle spaces and special characters
        section_uri = INST[f"section/{quote(section_id, safe='')}"]

        if section_type == 'form_section':
            self.graph.add((section_uri, RDF.type, DOC.FormSection))
        self.graph.add((section_uri, RDF.type, DOC.Section))

        self.graph.add((parent_uri, DOC.hasSection, section_uri))
        self.graph.add((section_uri, DOC.sectionId, Literal(section_id, datatype=XSD.string)))
        self.graph.add((section_uri, DOC.sectionType, Literal(section_type, datatype=XSD.string)))

        return section_uri

    def _create_field(self, parent_uri: URIRef, field_id: str, field_type: str) -> URIRef:
        """Create field instance"""
        # URL-encode the ID to handle spaces and special characters
        field_uri = INST[f"field/{quote(field_id, safe='')}"]

        if field_type == 'form_field':
            self.graph.add((field_uri, RDF.type, DOC.FormField))
        self.graph.add((field_uri, RDF.type, DOC.Field))

        self.graph.add((parent_uri, DOC.hasField, field_uri))
        self.graph.add((field_uri, DOC.fieldId, Literal(field_id, datatype=XSD.string)))
        self.graph.add((field_uri, DOC.fieldType, Literal(field_type, datatype=XSD.string)))

        return field_uri

    def _create_bounding_box(self, element_id: str, ann_data: Dict[str, Any]) -> URIRef:
        """Create bounding box node with normalized coordinates."""
        bbox_uri = INST[f"bbox/{quote(element_id, safe='')}"]
        self.graph.add((bbox_uri, RDF.type, BBOX.BoundingBox))

        if 'box_2d' in ann_data:
            box = ann_data['box_2d']
            self.graph.add((bbox_uri, BBOX.x1, Literal(box[0], datatype=XSD.integer)))
            self.graph.add((bbox_uri, BBOX.y1, Literal(box[1], datatype=XSD.integer)))
            self.graph.add((bbox_uri, BBOX.x2, Literal(box[2], datatype=XSD.integer)))
            self.graph.add((bbox_uri, BBOX.y2, Literal(box[3], datatype=XSD.integer)))

        if self.include_float_coords and 'x' in ann_data:
            self.graph.add((bbox_uri, BBOX.pixelX, Literal(ann_data['x'], datatype=XSD.float)))
            self.graph.add((bbox_uri, BBOX.pixelY, Literal(ann_data['y'], datatype=XSD.float)))
            self.graph.add((bbox_uri, BBOX.pixelWidth, Literal(ann_data['width'], datatype=XSD.float)))
            self.graph.add((bbox_uri, BBOX.pixelHeight, Literal(ann_data['height'], datatype=XSD.float)))

        return bbox_uri

    def _create_field_relationships(
        self,
        elements_by_id: Dict[str, URIRef],
        fields: Dict[str, URIRef],
        annotations: List[Dict[str, Any]]
    ):
        """Create relationships between fields and their labels/values"""
        # Build map of field_for relationships
        field_labels = {}  # field_for_id -> label_element_uri
        field_values = {}  # field_id -> value_element_uri

        for ann in annotations:
            element_id = ann.get('label', '')
            element_uri = elements_by_id.get(element_id)
            if not element_uri:
                continue

            # Check if this is a field label
            field_for = ann.get('field_for')
            if field_for:
                field_labels[field_for] = element_uri

            # Check if this is a field value
            if ann.get('entity_type') != 'FIELD_LABEL':
                # This is a value - find its field
                for field_id, field_uri in fields.items():
                    if element_id.startswith(field_id.replace('_field', '')):
                        field_values[field_id] = element_uri
                        break

        # Link labels and values to fields
        for field_id, field_uri in fields.items():
            # Find corresponding label
            value_base = field_id.replace('_field', '')
            label_uri = field_labels.get(value_base)
            if label_uri:
                self.graph.add((field_uri, DOC.hasLabel, label_uri))

            # Find corresponding value
            value_uri = field_values.get(field_id)
            if value_uri:
                self.graph.add((field_uri, DOC.hasValue, value_uri))

    def _get_entity_class(self, entity_type: str) -> URIRef:
        """Map entity type string to entity class URI (NOT annotation!)"""
        type_mapping = {
            'PERSON': DOC.Person,
            'ORG': DOC.Organization,
            'IDENTIFIER': DOC.Identifier,
            'DATE': DOC.Date,
            'MONEY': DOC.Money,
            'ADDRESS': DOC.Address,
            'PHONE_NUMBER': DOC.PhoneNumber,
        }
        return type_mapping.get(entity_type)

    def _add_linked_data_types(self, ann_uri: URIRef, entity_type: str):
        """Add schema.org and DBpedia type annotations"""
        schema_mapping = {
            'PERSON': SCHEMA.Person,
            'ORG': SCHEMA.Organization,
            'ADDRESS': SCHEMA.PostalAddress,
            'MONEY': SCHEMA.MonetaryAmount,
        }

        dbpedia_mapping = {
            'PERSON': DBO.Person,
            'ORG': DBO.Organisation,
            'ADDRESS': DBO.Address,
        }

        # Add schema.org type
        if entity_type in schema_mapping:
            self.graph.add((ann_uri, RDF.type, schema_mapping[entity_type]))

        # Add DBpedia type
        if entity_type in dbpedia_mapping:
            self.graph.add((ann_uri, RDF.type, dbpedia_mapping[entity_type]))

    def serialize(self, format: str = 'turtle') -> str:
        """
        Serialize graph to RDF format.

        Args:
            format: 'turtle', 'n3', 'json-ld', 'xml', 'nt'

        Returns:
            Serialized RDF string
        """
        return self.graph.serialize(format=format)

    def save(self, filepath: str, format: str = 'turtle'):
        """Save graph to file"""
        self.graph.serialize(destination=filepath, format=format)


def extract_document_structure(page_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract section, field, and annotation structure from page HTML/data.

    This function would be called by the pagedjs_renderer to enrich
    annotations with section/field metadata.
    """
    # This will be implemented in pagedjs_renderer
    # to extract data-section, data-field, data-field-for attributes
    pass
