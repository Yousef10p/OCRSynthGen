# Universal Document Structure Ontology — JSON Schema

Formal JSON Schema implementation of the 7-layer document structure ontology.

## Directory Structure

```
schemas/
├── v1.0/
│   ├── document.schema.json       # Main schema (references all layers)
│   ├── metadata.schema.json       # Layer 1: Document metadata
│   ├── header.schema.json         # Layer 2: Header
│   ├── content.schema.json        # Layer 3: Content blocks
│   ├── layout.schema.json         # Layer 4: Layout
│   ├── footer.schema.json         # Layer 5: Footer
│   ├── validation.schema.json     # Layer 6: Certifications & security
│   ├── relationships.schema.json  # Layer 7: Cross-document links
│   └── definitions.schema.json    # Shared types (BoundingBox, EntityBlock, etc.)
├── examples/
│   ├── invoice.json               # Commercial invoice example
│   ├── irs_1040.json              # IRS Form 1040 example
│   └── cms_1500.json              # CMS-1500 medical claim example
├── tools/
│   ├── validate.py                # CLI validation script
│   └── requirements.txt           # Python dependencies
└── README.md                      # This file
```

## Quick Start

### Validate a document

```bash
cd schemas/tools
pip install -r requirements.txt
python validate.py ../examples/invoice.json
```

### Validate all examples

```bash
python validate.py ../examples/
```

### Use in Python

```python
import json
from pathlib import Path
from jsonschema import validate, RefResolver

schema_dir = Path("schemas/v1.0")
with open(schema_dir / "document.schema.json") as f:
    schema = json.load(f)

# Build resolver for $ref resolution
store = {}
for sf in schema_dir.glob("*.schema.json"):
    with open(sf) as f:
        s = json.load(f)
        store[sf.name] = s

resolver = RefResolver(
    base_uri=f"file:///{schema_dir.resolve().as_posix()}/",
    referrer=schema,
    store=store,
)

validate(instance=your_document, schema=schema, resolver=resolver)
```

## 7-Layer Architecture

| Layer | Schema | Purpose |
|-------|--------|---------|
| 1 | `metadata.schema.json` | Document ID, type, archetype, category, language |
| 2 | `header.schema.json` | Title, issuer, recipient, dates, logo |
| 3 | `content.schema.json` | 30+ content block types (text, form fields, tables, signatures, etc.) |
| 4 | `layout.schema.json` | Page size, margins, columns, grid, flow direction |
| 5 | `footer.schema.json` | Signatures, totals, disclaimers, pagination |
| 6 | `validation.schema.json` | Certifications, security features, digital signatures |
| 7 | `relationships.schema.json` | Parent/child documents, references, amendments |

Only layers 1 (metadata) and 3 (content) are required. All others are optional.

## Relationship to DoCO-FD

These JSON Schemas are the **data format** counterpart to the **DoCO-FD OWL ontology** in `ontologies/doco-fd/`. Both describe the same document structure model:

- **DoCO-FD** (`ontologies/doco-fd/doco-fd.ttl`) — RDF/OWL classes and properties for semantic markup in HTML templates (RDFa)
- **JSON Schema** (`schemas/v1.0/`) — validation format for generated document annotations (JSON output)

The generator produces documents with DoCO-FD markup in HTML, then extracts structured JSON that conforms to these schemas.

## Version

- **Current:** 1.0.0
- **Status:** Initial release
- **Spec:** See `20260105-133425-document-ontology-specification.md` for the full design document
