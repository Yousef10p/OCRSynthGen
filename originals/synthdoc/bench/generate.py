"""
Dataset generation — HTML → PDF → Images + Labels.
Wraps synthdoc pipeline for notebook use.
"""

import json
import shutil
from pathlib import Path
from typing import List, Optional


def generate_dataset(
    output_dir: Path = Path("data"),
    templates: Optional[List[str]] = None,
    count: int = 10,
    keep_pdfs: bool = False,
    dpi: int = 96,
    output_formats: Optional[List[str]] = None,
) -> None:
    """
    Generate synthetic documents: HTML → Paged.js → PDF → JPG + JSON labels.

    Args:
        output_dir:   Where to write images/, labels/, html/
        templates:    Template names (None = all registered)
        count:        Documents per template
        keep_pdfs:    Save intermediate PDFs
        dpi:          Image resolution
        output_formats: Label formats (default: ["json"])
    """
    from synthdoc.templates.registry import registry
    from synthdoc.services.unified_generator import UnifiedDocumentGenerator
    from synthdoc.ner_data_generator import NERDataGenerator

    if output_formats is None:
        output_formats = ["json"]

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    html_dir = output_dir / "html"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    if keep_pdfs:
        (output_dir / "pdfs").mkdir(parents=True, exist_ok=True)

    if templates is None:
        templates = registry.list_templates()

    html_gen = UnifiedDocumentGenerator()
    ner_gen = NERDataGenerator(
        output_dir=str(output_dir),
        create_overlays=True,
        output_formats=output_formats,
        dpi=dpi,
    )

    doc_num = 0
    page_num = 0

    for template_name in templates:
        print(f"{template_name}...")
        for _ in range(count):
            try:
                result = html_gen.generate_document_with_template(template_name)
                html = result["html_content"]

                code = template_name.replace("_", "")[:3].upper()
                doc_id = f"{code}_{doc_num:06d}"

                ner_result = ner_gen.generate_dataset_sample(
                    template_html=html,
                    template_data={},
                    output_id=doc_id,
                    keep_pdf=keep_pdfs,
                )

                # Save HTML
                with open(html_dir / f"{doc_id}.html", "w", encoding="utf-8") as f:
                    f.write(html)

                if "pages" in ner_result:
                    for page_idx, page_result in enumerate(ner_result["pages"]):
                        page_id = f"{doc_id}_page_{page_idx + 1}"
                        final_img = images_dir / f"{page_id}.jpg"
                        final_label = labels_dir / f"{page_id}.json"

                        if Path(page_result["image_path"]) != final_img:
                            shutil.move(page_result["image_path"], final_img)
                        if Path(page_result["label_path"]) != final_label:
                            shutil.move(page_result["label_path"], final_label)

                        with open(final_label, "r", encoding="utf-8") as f:
                            anno = json.load(f)
                        anno["document_id"] = doc_id
                        anno["page_number"] = page_idx + 1
                        anno["total_pages"] = ner_result["total_pages"]
                        anno["template_type"] = template_name
                        with open(final_label, "w", encoding="utf-8") as f:
                            json.dump(anno, f, indent=2)

                        page_num += 1

                    if keep_pdfs and ner_result.get("pdf_path"):
                        shutil.move(ner_result["pdf_path"], output_dir / "pdfs" / f"{doc_id}.pdf")

                    print(f"  {doc_id} ({ner_result['total_pages']} pages)")
                else:
                    final_img = images_dir / f"{doc_id}.jpg"
                    final_label = labels_dir / f"{doc_id}.json"

                    if Path(ner_result["image_path"]) != final_img:
                        shutil.move(ner_result["image_path"], final_img)
                    if Path(ner_result["label_path"]) != final_label:
                        shutil.move(ner_result["label_path"], final_label)

                    with open(final_label, "r", encoding="utf-8") as f:
                        anno = json.load(f)
                    anno["document_id"] = doc_id
                    anno["template_type"] = template_name
                    with open(final_label, "w", encoding="utf-8") as f:
                        json.dump(anno, f, indent=2)

                    if keep_pdfs and ner_result.get("pdf_path"):
                        shutil.move(ner_result["pdf_path"], output_dir / "pdfs" / f"{doc_id}.pdf")

                    page_num += 1
                    print(f"  {doc_id}")

                doc_num += 1

            except Exception as e:
                import traceback
                print(f"  Error: {e}")
                traceback.print_exc()

    print(f"\nTotal: {doc_num} documents, {page_num} pages")
    print(f"Output: {output_dir}")
