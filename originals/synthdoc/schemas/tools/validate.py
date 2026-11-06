#!/usr/bin/env python3
"""
Document Ontology Validator

Validates document JSON against the Universal Document Structure Ontology schema.

Usage:
    python validate.py path/to/document.json
    python validate.py path/to/document.json --schema-dir ../v1.0
    python validate.py path/to/documents/ --verbose
"""

import json
import sys
from pathlib import Path

try:
    from jsonschema import validate, ValidationError, RefResolver
except ImportError:
    print("Error: jsonschema not installed. Run: pip install jsonschema")
    sys.exit(1)

try:
    import click
except ImportError:
    click = None


class DocumentValidator:
    def __init__(self, schema_dir: Path):
        self.schema_dir = schema_dir
        self.schemas = self._load_schemas()
        self.resolver = self._create_resolver()

    def _load_schemas(self) -> dict:
        schemas = {}
        for schema_file in self.schema_dir.glob("*.schema.json"):
            with open(schema_file) as f:
                schema = json.load(f)
                schemas[schema_file.name] = schema
        return schemas

    def _create_resolver(self) -> RefResolver:
        store = {}
        # Determine the base URL from the main schema's $id so relative $refs resolve
        main_schema = self.schemas.get("document.schema.json", {})
        main_id = main_schema.get("$id", "")
        base_url = main_id.rsplit("/", 1)[0] + "/" if "/" in main_id else ""

        for name, schema in self.schemas.items():
            schema_id = schema.get("$id", name)
            store[schema_id] = schema
            store[name] = schema
            # Also register under the URL that relative $refs will resolve to
            if base_url:
                store[base_url + name] = schema

        return RefResolver(
            base_uri=f"file:///{self.schema_dir.resolve().as_posix()}/",
            referrer=main_schema,
            store=store,
        )

    def validate_document(self, document: dict) -> tuple:
        """
        Validate a document dict against the ontology schema.

        Returns:
            (is_valid, errors) tuple
        """
        errors = []

        main_schema = self.schemas.get("document.schema.json")
        if not main_schema:
            errors.append({
                "path": [],
                "message": "document.schema.json not found in schema directory",
                "validator": "setup",
            })
            return False, errors

        try:
            validate(
                instance=document,
                schema=main_schema,
                resolver=self.resolver,
            )
        except ValidationError as e:
            errors.append({
                "path": list(e.absolute_path),
                "message": e.message,
                "validator": e.validator,
            })

        errors.extend(self._validate_semantics(document))
        return len(errors) == 0, errors

    def validate_file(self, file_path: Path) -> tuple:
        with open(file_path) as f:
            document = json.load(f)
        return self.validate_document(document)

    def _validate_semantics(self, document: dict) -> list:
        errors = []

        if "content" in document:
            block_ids = set()
            blocks = document["content"].get("blocks", [])

            for block in blocks:
                block_id = block.get("block_id")
                if block_id in block_ids:
                    errors.append({
                        "path": ["content", "blocks"],
                        "message": f"Duplicate block_id: {block_id}",
                        "validator": "semantic",
                    })
                block_ids.add(block_id)

                parent = block.get("parent_block")
                if parent and parent not in block_ids:
                    errors.append({
                        "path": ["content", "blocks", block_id],
                        "message": f"parent_block '{parent}' references unknown block",
                        "validator": "semantic",
                    })

            reading_order = document["content"].get("reading_order", [])
            for ref_id in reading_order:
                if ref_id not in block_ids:
                    errors.append({
                        "path": ["content", "reading_order"],
                        "message": f"reading_order references unknown block_id: {ref_id}",
                        "validator": "semantic",
                    })

        return errors


def main_cli():
    if len(sys.argv) < 2:
        print("Usage: python validate.py <document.json|directory> [--schema-dir DIR] [--verbose]")
        sys.exit(1)

    target = Path(sys.argv[1])
    schema_dir = Path(__file__).parent.parent / "v1.0"
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == "--schema-dir" and i + 1 < len(sys.argv):
            schema_dir = Path(sys.argv[i + 1])

    validator = DocumentValidator(schema_dir)

    files = []
    if target.is_dir():
        files = sorted(target.glob("*.json"))
    elif target.is_file():
        files = [target]
    else:
        print(f"Error: {target} not found")
        sys.exit(1)

    total = 0
    passed = 0
    failed = 0

    for file_path in files:
        total += 1
        is_valid, errors = validator.validate_file(file_path)

        if is_valid:
            passed += 1
            print(f"  PASS  {file_path.name}")
        else:
            failed += 1
            print(f"  FAIL  {file_path.name} ({len(errors)} error(s))")
            for i, error in enumerate(errors, 1):
                path_str = " > ".join(map(str, error["path"])) if error["path"] else "(root)"
                print(f"         [{i}] {path_str}: {error['message']}")
                if verbose:
                    print(f"             validator: {error['validator']}")

    print(f"\n{'=' * 40}")
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main_cli()
