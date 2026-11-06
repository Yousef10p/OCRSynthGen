"""
Data validation module for synthetic document generation.

Validates annotations, coordinates, entity types, and output consistency.
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"      # Blocks output
    WARNING = "warning"  # Logs but continues
    INFO = "info"        # Informational only


@dataclass
class ValidationIssue:
    """A single validation issue."""
    level: ValidationLevel
    message: str
    annotation_id: Optional[str] = None
    field_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "annotation_id": self.annotation_id,
            "field_name": self.field_name,
        }


@dataclass
class ValidationReport:
    """Complete validation report."""
    is_valid: bool
    errors: List[ValidationIssue]
    warnings: List[ValidationIssue]
    info: List[ValidationIssue]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.info),
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = []
        if self.is_valid:
            lines.append("[PASS] Validation PASSED")
        else:
            lines.append("[FAIL] Validation FAILED")

        lines.append(f"  Errors: {len(self.errors)}")
        lines.append(f"  Warnings: {len(self.warnings)}")
        lines.append(f"  Info: {len(self.info)}")

        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors[:5]:  # Show first 5
                lines.append(f"  - {err.message}")
            if len(self.errors) > 5:
                lines.append(f"  ... and {len(self.errors) - 5} more")

        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings[:3]:  # Show first 3
                lines.append(f"  - {warn.message}")
            if len(self.warnings) > 3:
                lines.append(f"  ... and {len(self.warnings) - 3} more")

        return "\n".join(lines)


class AnnotationValidator:
    """Validates annotation data."""

    VALID_ENTITY_TYPES = {
        'PERSON', 'ORG', 'IDENTIFIER', 'DATE', 'MONEY',
        'ADDRESS', 'PHONE_NUMBER', 'FIELD_LABEL'
    }

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, warnings become errors
        """
        self.strict = strict

    def validate_annotations(
        self,
        annotations: List[Dict[str, Any]],
        image_width: int = 1200,
        image_height: int = 1555
    ) -> ValidationReport:
        """
        Validate list of annotations.

        Args:
            annotations: List of annotation dictionaries
            image_width: Expected image width for coordinate validation
            image_height: Expected image height for coordinate validation

        Returns:
            ValidationReport with all issues found
        """
        issues: List[ValidationIssue] = []

        for idx, ann in enumerate(annotations):
            ann_id = ann.get('label', f'annotation_{idx}')

            # Required fields
            if not ann.get('text_content'):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message=f"Missing text_content",
                    annotation_id=ann_id
                ))

            if not ann.get('entity_type'):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message=f"Missing entity_type",
                    annotation_id=ann_id
                ))
            elif ann['entity_type'] not in self.VALID_ENTITY_TYPES:
                level = ValidationLevel.ERROR if self.strict else ValidationLevel.WARNING
                issues.append(ValidationIssue(
                    level=level,
                    message=f"Invalid entity_type: {ann['entity_type']}",
                    annotation_id=ann_id
                ))

            # Bounding box validation
            if 'box_2d' in ann:
                bbox_issues = self._validate_bbox(
                    ann['box_2d'],
                    ann_id,
                    image_width,
                    image_height
                )
                issues.extend(bbox_issues)
            else:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"Missing bounding box (box_2d)",
                    annotation_id=ann_id
                ))

        # Check for overlaps
        overlap_issues = self._check_overlaps(annotations)
        issues.extend(overlap_issues)

        # Separate by level
        errors = [i for i in issues if i.level == ValidationLevel.ERROR]
        warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
        info_items = [i for i in issues if i.level == ValidationLevel.INFO]

        return ValidationReport(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            info=info_items
        )

    def _validate_bbox(
        self,
        box: List[int],
        ann_id: str,
        img_width: int,
        img_height: int
    ) -> List[ValidationIssue]:
        """Validate bounding box coordinates."""
        issues = []

        if len(box) != 4:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"box_2d must have 4 coordinates, got {len(box)}",
                annotation_id=ann_id
            ))
            return issues

        x1, y1, x2, y2 = box

        # Check coordinate order
        if x1 >= x2:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Invalid box: x1 ({x1}) >= x2 ({x2})",
                annotation_id=ann_id
            ))

        if y1 >= y2:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Invalid box: y1 ({y1}) >= y2 ({y2})",
                annotation_id=ann_id
            ))

        # Check bounds (normalized 0-1000)
        if not (0 <= x1 <= 1000 and 0 <= x2 <= 1000):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"x coordinates out of bounds [0-1000]: ({x1}, {x2})",
                annotation_id=ann_id
            ))

        if not (0 <= y1 <= 1000 and 0 <= y2 <= 1000):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"y coordinates out of bounds [0-1000]: ({y1}, {y2})",
                annotation_id=ann_id
            ))

        # Check box size (detect suspiciously small boxes)
        width = x2 - x1
        height = y2 - y1

        if width < 5 or height < 5:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message=f"Suspiciously small bounding box: {width}x{height} normalized units",
                annotation_id=ann_id
            ))

        return issues

    def _check_overlaps(
        self,
        annotations: List[Dict[str, Any]]
    ) -> List[ValidationIssue]:
        """Check for overlapping bounding boxes."""
        issues = []

        # Get annotations with valid boxes
        valid_anns = [
            ann for ann in annotations
            if 'box_2d' in ann and len(ann['box_2d']) == 4
        ]

        # Compare each pair
        for i, ann1 in enumerate(valid_anns):
            for ann2 in valid_anns[i+1:]:
                overlap_pct = self._calculate_overlap(
                    ann1['box_2d'],
                    ann2['box_2d']
                )

                if overlap_pct > 0.8:  # >80% overlap
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"High overlap ({overlap_pct:.1%}) with {ann2.get('label')}",
                        annotation_id=ann1.get('label')
                    ))

        return issues

    def _calculate_overlap(self, box1: List[int], box2: List[int]) -> float:
        """Calculate overlap percentage between two boxes (IoU)."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap

        if intersection == 0:
            return 0.0

        # Calculate areas
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class OutputValidator:
    """Validates output files for consistency."""

    def validate_output_consistency(
        self,
        json_path: str,
        ttl_path: Optional[str] = None,
        jsonld_path: Optional[str] = None
    ) -> ValidationReport:
        """
        Validate consistency between output formats.

        Args:
            json_path: Path to JSON file (required)
            ttl_path: Path to Turtle file (optional)
            jsonld_path: Path to JSON-LD file (optional)

        Returns:
            ValidationReport with consistency issues
        """
        issues = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Invalid JSON: {str(e)}"
            ))
            return ValidationReport(is_valid=False, errors=issues, warnings=[], info=[])

        json_ann_count = len(json_data.get('annotations', []))

        # Check TTL if provided
        if ttl_path:
            ttl_file = Path(ttl_path)
            if ttl_file.exists():
                ttl_issues = self._validate_ttl_consistency(ttl_file, json_ann_count)
                issues.extend(ttl_issues)
            else:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"TTL file not found: {ttl_path}"
                ))

        # Check JSON-LD if provided
        if jsonld_path:
            jsonld_file = Path(jsonld_path)
            if not jsonld_file.exists():
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"JSON-LD file not found: {jsonld_path}"
                ))

        errors = [i for i in issues if i.level == ValidationLevel.ERROR]
        warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
        info_items = [i for i in issues if i.level == ValidationLevel.INFO]

        return ValidationReport(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            info=info_items
        )

    def _validate_ttl_consistency(self, ttl_path: Path, expected_count: int) -> List[ValidationIssue]:
        """Validate TTL file consistency with JSON."""
        issues = []

        try:
            from rdflib import Graph

            g = Graph()
            g.parse(ttl_path, format='turtle')

            # Count text elements
            query = """
            PREFIX doc: <http://example.org/document-ontology#>
            SELECT (COUNT(?elem) as ?count)
            WHERE {
                ?elem a doc:TextElement .
            }
            """

            result = list(g.query(query))
            if result:
                ttl_count = int(result[0][0])

                if ttl_count != expected_count:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"Element count mismatch: JSON has {expected_count}, TTL has {ttl_count}"
                    ))

        except ImportError:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                message="rdflib not available, skipping TTL validation"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Error parsing TTL file: {str(e)}"
            ))

        return issues


class TemplateValidator:
    """Validates templates before generation."""

    def validate_template(self, template_class) -> ValidationReport:
        """
        Validate a template class before using it for generation.

        Args:
            template_class: Template class to validate

        Returns:
            ValidationReport with validation results
        """
        issues = []

        # Check if template has required methods
        if not hasattr(template_class, 'get_template_string'):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message="Template missing required method: get_template_string()"
            ))

        if not hasattr(template_class, 'generate_data'):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Template missing generate_data() method"
            ))

        # Try instantiating the template
        try:
            template_instance = template_class()
        except Exception as e:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Failed to instantiate template: {str(e)}"
            ))
            errors = [i for i in issues if i.level == ValidationLevel.ERROR]
            warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
            return ValidationReport(is_valid=False, errors=errors, warnings=warnings, info=[])

        # Try getting template string
        try:
            template_str = template_instance.get_template_string()
            if not template_str or not isinstance(template_str, str):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message="get_template_string() returned empty or non-string value"
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"get_template_string() failed: {str(e)}"
            ))

        # Try generating test data
        try:
            if hasattr(template_instance, 'generate_data'):
                test_data = template_instance.generate_data()
                if not isinstance(test_data, dict):
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message="generate_data() should return a dictionary"
                    ))
        except Exception as e:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message=f"generate_data() failed: {str(e)}"
            ))

        # Check for data-label attributes in template
        try:
            template_str = template_instance.get_template_string()
            if 'data-label=' not in template_str and 'data-entity-type=' not in template_str:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message="Template contains no data-label or data-entity-type attributes for NER extraction"
                ))
        except:
            pass

        errors = [i for i in issues if i.level == ValidationLevel.ERROR]
        warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
        info_items = [i for i in issues if i.level == ValidationLevel.INFO]

        return ValidationReport(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            info=info_items
        )


# Convenience functions for users

def validate_template(template_class) -> ValidationReport:
    """
    Convenience function to validate a template class.

    Args:
        template_class: Template class to validate

    Returns:
        ValidationReport

    Example:
        >>> from synthdoc.templates.invoice_template import InvoiceTemplate
        >>> report = validate_template(InvoiceTemplate)
        >>> if not report.is_valid:
        ...     print(f"Template validation failed: {report.error_count} errors")
    """
    validator = TemplateValidator()
    return validator.validate_template(template_class)


def validate_page_data(page_data: Dict[str, Any], strict: bool = False) -> ValidationReport:
    """
    Convenience function to validate page data.

    Args:
        page_data: Page data dictionary with 'annotations' key
        strict: If True, warnings become errors

    Returns:
        ValidationReport

    Example:
        >>> report = validate_page_data(page_data)
        >>> if not report.is_valid:
        ...     print(f"Validation failed: {report.error_count} errors")
        ...     for error in report.errors:
        ...         print(f"  - {error.message}")
    """
    validator = AnnotationValidator(strict=strict)
    annotations = page_data.get('annotations', [])
    return validator.validate_annotations(annotations)


def validate_output_files(base_path: str, document_id: str, page_num: int = 1) -> ValidationReport:
    """
    Convenience function to validate all output files for a document page.

    Args:
        base_path: Base directory containing 'labels' subdirectory
        document_id: Document identifier
        page_num: Page number

    Returns:
        ValidationReport

    Example:
        >>> report = validate_output_files("data", "DOC_000001", 1)
        >>> print(report)
    """
    labels_dir = Path(base_path) / "labels"
    # Try flat Bates naming first, fall back to legacy _page_ naming
    json_path = labels_dir / f"{document_id}.json"
    if not json_path.exists():
        json_path = labels_dir / f"{document_id}_page_{page_num}.json"
    ttl_path = labels_dir / f"{document_id}.ttl"
    if not ttl_path.exists():
        ttl_path = labels_dir / f"{document_id}_page_{page_num}.ttl"
    jsonld_path = labels_dir / f"{document_id}.jsonld"
    if not jsonld_path.exists():
        jsonld_path = labels_dir / f"{document_id}_page_{page_num}.jsonld"

    validator = OutputValidator()
    return validator.validate_output_consistency(
        str(json_path),
        str(ttl_path) if ttl_path.exists() else None,
        str(jsonld_path) if jsonld_path.exists() else None
    )
