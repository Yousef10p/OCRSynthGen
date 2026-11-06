#!/usr/bin/env python3
"""
Generate a dataset using the HTML -> PDF -> Image pipeline.

Installed as the ``synthdoc`` command; also callable as ``synthdoc.generate()``.
"""
from pathlib import Path
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from synthdoc.templates.registry import registry
from synthdoc.services.unified_generator import UnifiedDocumentGenerator
from synthdoc.ner_data_generator import NERDataGenerator
from synthdoc.config import load_config, SyntheticDocsConfig


def _next_bates_number(data_dir: Path, count: int = 1) -> int:
    """Read and atomically increment the Bates counter for a data directory.

    Returns the starting number. The counter file is incremented by `count`.
    Format: SDD{NNNNNN} — globally unique sequential IDs per page.
    """
    counter_file = data_dir / ".bates_counter"
    start = 0
    if counter_file.exists():
        try:
            start = int(counter_file.read_text().strip())
        except (ValueError, OSError):
            # Fallback: scan existing labels to find the max
            labels_dir = data_dir / "labels"
            if labels_dir.exists():
                for lf in labels_dir.glob("*.json"):
                    stem = lf.stem.rsplit("_page_", 1)[0]  # legacy compat
                    # Extract number from SDD{NNNNNN}, flat SDD{NNNNNN}, or legacy {CODE}_{NNNNNN}
                    if stem.startswith("SDD"):
                        try:
                            num = int(stem[3:])
                            start = max(start, num + 1)
                            continue
                        except ValueError:
                            pass
                    parts = stem.rsplit("_", 1) if "_" in stem else [stem]
                    try:
                        num = int(parts[-1])
                        start = max(start, num + 1)
                    except (ValueError, IndexError):
                        pass
    counter_file.write_text(str(start + count))
    return start



def _generate_template_batch(template_name, count, data_dir,
                             keep_pdfs, create_test_output, entity_types,
                             min_bbox_size, output_formats, config):
    """Worker function: generate all documents for a single template using temp IDs.

    Each worker creates its own UnifiedDocumentGenerator and NERDataGenerator
    (and thus its own Playwright browser) so threads don't share mutable state.

    Returns (template_name, doc_results) where doc_results is a list of dicts
    containing temp file paths and page counts for later Bates assignment.
    """
    import traceback as _tb

    html_gen = UnifiedDocumentGenerator()
    ner_gen = NERDataGenerator(
        output_dir=str(data_dir),
        create_overlays=True,
        create_test_output=create_test_output,
        entity_types=entity_types,
        min_bbox_size=min_bbox_size,
        bbox_padding=config.generation.bbox_padding,
        filter_tiny_boxes=config.generation.filter_tiny_boxes,
        output_formats=output_formats,
        images_subdir=config.output.images_subdir,
        labels_subdir=config.output.labels_subdir,
        pdfs_subdir=config.output.pdfs_subdir,
        dpi=config.generation.dpi
    )

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    html_dir = data_dir / "html"
    overlays_dir = data_dir / "overlays"

    doc_results = []

    # Get template instance for ontology metadata (if ontology output enabled)
    tpl_instance = None
    if 'ontology' in output_formats:
        try:
            tpl_class = registry.get_template(template_name)
            tpl_instance = tpl_class()
        except Exception:
            pass  # ontology output will be skipped for this template

    print(f"{template_name}...")

    for i in range(count):
        try:
            result = html_gen.generate_document_with_template(template_name)
            html_content = result['html_content']

            temp_id = f"_tmp_{template_name}_{i}"

            ner_result = ner_gen.generate_dataset_sample(
                template_html=html_content,
                template_data={},
                output_id=temp_id,
                keep_pdf=keep_pdfs,
                template_instance=tpl_instance
            )

            # Save HTML with temp name
            html_path = html_dir / f"{temp_id}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Collect page file paths
            pages_info = []
            if 'pages' in ner_result:
                for page_idx, page_result in enumerate(ner_result['pages']):
                    page_num = page_idx + 1
                    temp_page_id = f"{temp_id}_page_{page_num}"

                    # Ensure files are at expected temp locations
                    temp_img = images_dir / f"{temp_page_id}.jpg"
                    temp_label = labels_dir / f"{temp_page_id}.json"

                    if Path(page_result['image_path']) != temp_img:
                        shutil.move(page_result['image_path'], temp_img)
                    if Path(page_result['label_path']) != temp_label:
                        shutil.move(page_result['label_path'], temp_label)

                    # Check for overlay
                    temp_overlay = overlays_dir / f"{temp_page_id}_overlay.jpg"
                    if not temp_overlay.exists():
                        temp_overlay = None
                    else:
                        temp_overlay = str(temp_overlay)

                    pages_info.append({
                        'temp_image': str(temp_img),
                        'temp_label': str(temp_label),
                        'temp_overlay': temp_overlay,
                        'page_number': page_num,
                    })

                total_pages = ner_result['total_pages']
                print(f"  {temp_id} ({total_pages} pages)")
            else:
                temp_img = images_dir / f"{temp_id}.jpg"
                temp_label = labels_dir / f"{temp_id}.json"

                if Path(ner_result['image_path']) != temp_img:
                    shutil.move(ner_result['image_path'], temp_img)
                if Path(ner_result['label_path']) != temp_label:
                    shutil.move(ner_result['label_path'], temp_label)

                temp_overlay = overlays_dir / f"{temp_id}_overlay.jpg"
                if not temp_overlay.exists():
                    temp_overlay = None
                else:
                    temp_overlay = str(temp_overlay)

                pages_info.append({
                    'temp_image': str(temp_img),
                    'temp_label': str(temp_label),
                    'temp_overlay': temp_overlay,
                    'page_number': 1,
                })
                total_pages = 1
                print(f"  {temp_id}")

            # Collect PDF path if present
            temp_pdf = None
            if keep_pdfs and ner_result.get('pdf_path'):
                temp_pdf = ner_result['pdf_path']

            doc_results.append({
                'template_name': template_name,
                'temp_id': temp_id,
                'page_count': total_pages,
                'pages': pages_info,
                'temp_pdf': temp_pdf,
                'temp_html': str(html_path),
            })

        except Exception as e:
            print(f"  Error [{template_name}]: {e}")
            _tb.print_exc()

    return template_name, doc_results


def _assign_bates_and_rename(all_results, data_dir, keep_pdfs):
    """Post-render: assign sequential per-page Bates numbers and rename all files.

    Args:
        all_results: List of (template_name, doc_results) from workers.
                     Iterated in deterministic template order.
        data_dir: Path to data directory
        keep_pdfs: Whether PDFs were generated

    Returns:
        (manifest_dict, total_docs, total_pages)
    """
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    html_dir = data_dir / "html"
    overlays_dir = data_dir / "overlays"
    pdfs_dir = data_dir / "pdfs" if keep_pdfs else None

    # Count total pages across all results
    total_pages = sum(
        doc['page_count']
        for _, doc_list in all_results
        for doc in doc_list
    )

    if total_pages == 0:
        return {"documents": []}, 0, 0

    # Reserve Bates numbers for all pages at once
    bates_start = _next_bates_number(data_dir, total_pages)

    manifest = {"documents": []}
    current_bates = bates_start
    total_docs = 0

    for template_name, doc_list in all_results:
        for doc in doc_list:
            doc_bates_start = current_bates
            page_bates_ids = []

            for page_info in doc['pages']:
                bates_id = f"SDD{current_bates:06d}"
                page_bates_ids.append(bates_id)
                page_num = page_info['page_number']

                # Rename image
                final_img = images_dir / f"{bates_id}.jpg"
                src_img = Path(page_info['temp_image'])
                if src_img.exists():
                    shutil.move(str(src_img), str(final_img))

                # Rename and rewrite label JSON
                final_label = labels_dir / f"{bates_id}.json"
                src_label = Path(page_info['temp_label'])
                if src_label.exists():
                    with open(src_label, 'r', encoding='utf-8') as f:
                        anno = json.load(f)
                    anno['document_id'] = bates_id
                    anno['document_group'] = {
                        'bates_start': f"SDD{doc_bates_start:06d}",
                        'bates_end': f"SDD{doc_bates_start + doc['page_count'] - 1:06d}",
                        'template': template_name,
                    }
                    anno['page_number'] = page_num
                    anno['total_pages'] = doc['page_count']
                    anno['template_type'] = template_name
                    with open(final_label, 'w', encoding='utf-8') as f:
                        json.dump(anno, f, indent=2, ensure_ascii=False)
                    # Remove temp label if it's a different path
                    if src_label != final_label and src_label.exists():
                        src_label.unlink(missing_ok=True)

                # Rename RDF and ontology files (.ttl, .jsonld, .ontology.json) if they exist
                temp_stem = src_label.stem  # e.g., "_tmp_bank_statement_0_page_1"
                for rdf_ext in ['.ttl', '.jsonld', '.ontology.json']:
                    src_rdf = labels_dir / f"{temp_stem}{rdf_ext}"
                    if src_rdf.exists():
                        final_rdf = labels_dir / f"{bates_id}{rdf_ext}"
                        shutil.move(str(src_rdf), str(final_rdf))

                # Rename overlay if present
                if page_info.get('temp_overlay'):
                    src_overlay = Path(page_info['temp_overlay'])
                    if src_overlay.exists():
                        final_overlay = overlays_dir / f"{bates_id}_overlay.jpg"
                        shutil.move(str(src_overlay), str(final_overlay))

                current_bates += 1

            # Clean up temp overlay PDF (generated by pagedjs_renderer)
            temp_overlay_pdf = overlays_dir / f"{doc['temp_id']}_overlay.pdf"
            if temp_overlay_pdf.exists():
                if len(page_bates_ids) == 1:
                    final_overlay_pdf = overlays_dir / f"{page_bates_ids[0]}_overlay.pdf"
                else:
                    final_overlay_pdf = overlays_dir / f"{page_bates_ids[0]}-{page_bates_ids[-1]}_overlay.pdf"
                shutil.move(str(temp_overlay_pdf), str(final_overlay_pdf))

            # Rename PDF
            pdf_name = None
            if keep_pdfs and doc.get('temp_pdf'):
                src_pdf = Path(doc['temp_pdf'])
                if src_pdf.exists():
                    if len(page_bates_ids) == 1:
                        pdf_name = f"{page_bates_ids[0]}.pdf"
                    else:
                        pdf_name = f"{page_bates_ids[0]}-{page_bates_ids[-1]}.pdf"
                    shutil.move(str(src_pdf), str(pdfs_dir / pdf_name))

            # Rename HTML (use first page's Bates ID)
            if doc.get('temp_html'):
                src_html = Path(doc['temp_html'])
                if src_html.exists():
                    final_html = html_dir / f"{page_bates_ids[0]}.html"
                    shutil.move(str(src_html), str(final_html))

            manifest["documents"].append({
                "template": template_name,
                "bates_start": page_bates_ids[0],
                "bates_end": page_bates_ids[-1],
                "pages": page_bates_ids,
                "pdf": pdf_name,
            })

            total_docs += 1

    # Write manifest
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest, total_docs, total_pages


def generate(templates=None, count=None, keep_pdfs=None, create_test_output=False,
             config=None, entity_types=None, output_formats=None,
             min_bbox_size=None, validate_templates=False, parallel_workers=None,
             out_dir=None):
    """
    Generate dataset using HTML -> Paged.js -> Images pipeline

    Args:
        templates: List of template names (None = all from config/registry)
        count: Documents per template (None = use config default)
        keep_pdfs: Whether to save PDF files (None = use config default)
        create_test_output: Whether to create test output with metrics and diagnostics
        config: SyntheticDocsConfig instance (None = load from files/env)
        entity_types: List of entity types to include (None = all or from config)
        output_formats: List of output formats (None = use config default)
        min_bbox_size: Minimum bbox size (None = use config default)
        validate_templates: Whether to validate templates before generation
        parallel_workers: Number of parallel workers (None = use config default)
        out_dir: Output directory (None = config output.base_dir, relative to the working directory)

    Returns:
        The manifest dict also written to ``out_dir``: one entry per document with its Bates range.
    """
    # Load config if not provided
    if config is None:
        config = load_config()

    # Apply config defaults
    if count is None:
        count = config.generation.count_per_template
    if keep_pdfs is None:
        keep_pdfs = config.output.keep_pdfs
    if entity_types is None:
        entity_types = config.generation.entity_types
    if output_formats is None:
        output_formats = config.output.formats
    if min_bbox_size is None:
        min_bbox_size = config.generation.min_bbox_size
    if templates is None:
        templates = config.generation.templates
    if parallel_workers is None:
        parallel_workers = config.generation.parallel_workers

    data_dir = Path(out_dir) if out_dir is not None else Path(config.output.base_dir)

    # Regular dataset generation
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    html_dir = data_dir / "html"
    pdfs_dir = data_dir / "pdfs" if keep_pdfs else None

    # Create directories (don't delete existing files)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    if keep_pdfs:
        pdfs_dir.mkdir(parents=True, exist_ok=True)

    # Get templates
    if templates is None:
        templates = registry.list_templates()

    # Validate templates if requested
    if validate_templates and config.validation.validate_before_generation:
        print("Validating templates...")
        failed_templates = []
        for template_name in templates:
            try:
                report = registry.validate(template_name)
                if not report.is_valid:
                    print(f"  {template_name}: FAILED ({len(report.errors)} errors)")
                    for err in report.errors[:3]:
                        print(f"    - {err.message}")
                    failed_templates.append(template_name)
            except Exception as e:
                print(f"  {template_name}: ERROR - {e}")
                failed_templates.append(template_name)

        if failed_templates:
            templates = [t for t in templates if t not in failed_templates]
            print(f"\nSkipping {len(failed_templates)} templates that failed validation; "
                  f"proceeding with {len(templates)}\n")

    print("Using Paged.js for automatic pagination")
    if create_test_output:
        print("Test output enabled - will generate diagnostic images and PDFs")

    # Determine effective worker count
    workers = min(parallel_workers, len(templates))

    # Phase 1: Render all documents with temp IDs (parallelizable)
    all_results = []  # List of (template_name, doc_results) in deterministic order

    if workers <= 1:
        # Sequential generation
        for template_name in templates:
            result = _generate_template_batch(
                template_name, count,
                data_dir, keep_pdfs, create_test_output, entity_types,
                min_bbox_size, output_formats, config
            )
            all_results.append(result)
    else:
        # Parallel generation
        print(f"Parallel mode: {workers} workers for {len(templates)} templates")

        # Collect results keyed by template for deterministic ordering
        results_by_template = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _generate_template_batch,
                    tpl, count,
                    data_dir, keep_pdfs, create_test_output, entity_types,
                    min_bbox_size, output_formats, config
                ): tpl
                for tpl in templates
            }

            for future in as_completed(futures):
                tpl = futures[future]
                try:
                    result = future.result()
                    results_by_template[tpl] = result
                except Exception as e:
                    print(f"  Worker error [{tpl}]: {e}")
                    results_by_template[tpl] = (tpl, [])

        # Maintain deterministic template order for Bates assignment
        all_results = [results_by_template[tpl] for tpl in templates if tpl in results_by_template]

    # Phase 2: Assign per-page Bates numbers and rename files (sequential)
    print("\nAssigning Bates numbers...")
    manifest, doc_num, page_num = _assign_bates_and_rename(all_results, data_dir, keep_pdfs)

    print(f"\nTotal: {doc_num} documents, {page_num} pages")
    print(f"Output: {data_dir}")
    if manifest.get("documents"):
        first = manifest["documents"][0]["bates_start"]
        last = manifest["documents"][-1]["bates_end"]
        print(f"Bates range: {first} - {last}")

    return manifest


def main():
    """Main entry point for CLI."""
    import argparse
    import sys
    from pathlib import Path
    from datetime import datetime

    parser = argparse.ArgumentParser(prog="synthdoc", description="Generate synthetic documents with pixel-accurate entity annotations")
    parser.add_argument("--list", action="store_true", help="List available templates and exit")
    parser.add_argument("--out", type=str, help="Output directory (default: config output.base_dir, i.e. ./data)")
    parser.add_argument("--template-dir", action="append", default=[], metavar="DIR",
                        help="Also load templates from DIR, e.g. a clone of synthdoc-templates (repeatable)")
    parser.add_argument("--templates", nargs="+", help="Specific templates to generate (default: all)")
    parser.add_argument("--count", type=int, help="Documents per template (default: from config or 10)")
    parser.add_argument("--no-pdfs", action="store_true", help="Don't save PDF files")
    parser.add_argument("--test-output", action="store_true", help="Create test output with metrics and diagnostics")
    parser.add_argument("--config", type=str, help="Path to config file (YAML/JSON)")
    parser.add_argument("--entity-types", nargs="+", help="Entity types to include (e.g., PERSON ORG DATE)")
    parser.add_argument("--formats", nargs="+", choices=["json", "ttl", "jsonld"], help="Output formats (default: json ttl jsonld)")
    parser.add_argument("--min-bbox-size", type=int, help="Minimum bbox size in normalized units")
    parser.add_argument("--validate", action="store_true", help="Validate templates before generation")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default: 1 = sequential)")
    args = parser.parse_args()

    from synthdoc.templates.discovery import add_template_dir
    for template_dir in args.template_dir:
        add_template_dir(template_dir)

    if args.list:
        for category in sorted(registry.get_categories()):
            for name in sorted(registry.list_templates(category)):
                print(f"{category}\t{name}")
        return

    # Load config
    config = load_config(config_path=args.config) if args.config else None

    # Setup automatic log file saving to data directory
    if args.out:
        data_dir = Path(args.out)
    else:
        data_dir = Path((config or load_config()).output.base_dir)

    # Create logs directory in data folder
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"generation_{timestamp}.log"

    # Tee stdout/stderr to log file
    class TeeLogger:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    # Setup Python logging with proper levels
    import logging

    # Create logger
    logger = logging.getLogger('generate_dataset')
    logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Generation log: {log_file}")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"DPI setting: {config.generation.dpi if config else 'default (96)'}")
    logger.info("")

    generate(
        templates=args.templates,
        count=args.count,
        keep_pdfs=None if not args.no_pdfs else False,
        create_test_output=args.test_output,
        config=config,
        entity_types=args.entity_types,
        output_formats=args.formats,
        min_bbox_size=args.min_bbox_size,
        validate_templates=args.validate,
        parallel_workers=args.workers,
        out_dir=data_dir,
    )


if __name__ == "__main__":
    main()
