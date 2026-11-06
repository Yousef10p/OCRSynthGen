"""
Live model inference — requires GPU.
GLM-OCR, Qwen3-VL OCR, DeepSeek OCR, Qwen3 NER, VLM segmentation.
"""

import json
import re
from typing import Callable, Dict, List, Optional

from .helpers import STRUCTURAL_TYPES, parse_ner_response, parse_seg_response

NER_PROMPT = (
    "Extract all named entities from the following text.\n"
    "For each entity, also provide a confidence score (0.0 to 1.0).\n"
    "Return the results as a JSON array with this structure:\n"
    '[  {"text": "entity text", "type": "PERSON|ORG|DATE|ADDRESS|AMOUNT|PHONE|ID|LOCATION", "confidence": 0.95}\n]\n\n'
    "Text: {text}\n\n"
    "Return ONLY the JSON array, no other text."
)

SEG_PROMPT = """Identify all document regions in this image. For each region, provide:
- type: one of FIELD_LABEL, TABLE_CELL, TABLE_HEADER, TITLE, SECTION_TITLE, TEXT, CAPTION
- box: bounding box as [x1, y1, x2, y2] in pixel coordinates

The image is {width}x{height} pixels.

Return ONLY a JSON array:
[{{"type": "FIELD_LABEL", "box": [10, 20, 200, 40]}}, ...]"""


# ── Qwen3-VL OCR ────────────────────────────────────────────────────────

def run_qwenvl_ocr(pages: List[Dict], model_id: Optional[str] = None) -> None:
    """Run Qwen3-VL OCR on pages. Writes 'qwenvl_ocr' key in-place."""
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from PIL import Image

    mid = model_id or "Qwen/Qwen3-VL-8B-Instruct"
    processor = AutoProcessor.from_pretrained(mid)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map="auto"
    )

    for i, page in enumerate(pages):
        img = Image.open(page["image_path"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Extract ALL text from this document image exactly as written. Preserve reading order and line breaks. Return only the extracted text."},
        ]}]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_input], images=[img], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        page["qwenvl_ocr"] = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        if (i + 1) % 10 == 0:
            print(f"  Qwen-VL OCR: {i + 1}/{len(pages)}")

    del model, processor
    torch.cuda.empty_cache()
    print(f"Qwen3-VL OCR done — {len(pages)} pages")


# ── DeepSeek OCR ─────────────────────────────────────────────────────────

def _patch_llama_compat():
    """Patch transformers for DeepSeek-OCR-2 compat with transformers>=4.46.

    DeepSeek's custom code may trigger imports of removed Llama attention
    classes. We no-op the stubs since DeepSeek uses its own attention.

    Also patches DynamicCache attributes removed in newer transformers
    (seen_tokens, get_max_length, get_usable_length).
    """
    import transformers.models.llama.modeling_llama as llama_mod
    for cls_name in ("LlamaFlashAttention2", "LlamaSdpaAttention"):
        if not hasattr(llama_mod, cls_name):
            setattr(llama_mod, cls_name, llama_mod.LlamaAttention)

    from transformers.utils import import_utils
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False

    from transformers.cache_utils import DynamicCache
    for name, impl in {
        "seen_tokens": property(lambda self: self.get_seq_length()),
        "get_max_length": lambda self: None,
        "get_usable_length": lambda self, new_tokens=1, layer_idx=0: self.get_seq_length(layer_idx),
    }.items():
        if not hasattr(DynamicCache, name):
            setattr(DynamicCache, name, impl)


def _clean_deepseek_stdout(raw: str) -> str:
    """Strip debug noise from DeepSeek stdout capture.

    Removes tensor shape debug lines (BASE:/PATCHES:/===) and
    any leftover grounding tags (<|ref|>, <|det|>).
    """
    lines = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip tensor debug lines
        if stripped.startswith(("=", "BASE:", "PATCHES:")):
            continue
        # Strip grounding tags if present
        line = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", line)
        line = re.sub(r"<\|det\|>.*?<\|/det\|>", "", line)
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def run_deepseek_ocr(pages: List[Dict], on_progress: Optional[Callable] = None, model_id: Optional[str] = None) -> None:
    """Run DeepSeek-OCR-2 on pages. Writes 'deepseek_ocr' key in-place."""
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    _patch_llama_compat()

    mid = model_id or "deepseek-ai/DeepSeek-OCR-2"
    tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)

    config = AutoConfig.from_pretrained(mid, trust_remote_code=True)
    if hasattr(config, 'attn_implementation'):
        config.attn_implementation = 'eager'

    model = AutoModel.from_pretrained(
        mid, config=config, trust_remote_code=True, use_safetensors=True
    ).eval()

    if torch.cuda.is_available():
        model = model.cuda().to(torch.bfloat16)

    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")

    import io, sys
    for i, page in enumerate(pages):
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            model.infer(
                tokenizer,
                prompt="<image>\nConvert the document to markdown.",
                image_file=page["image_path"],
                output_path=tmp_dir,
                base_size=1024, image_size=768, crop_mode=True,
            )
        finally:
            sys.stdout = old_stdout
        page["deepseek_ocr"] = _clean_deepseek_stdout(buf.getvalue())
        if i == 0:
            print(f"  [DEBUG] page 0 captured {len(page['deepseek_ocr'])} chars")
        if (i + 1) % 5 == 0:
            print(f"  DeepSeek OCR: {i + 1}/{len(pages)}")
            if on_progress:
                on_progress()

    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"DeepSeek OCR done — {len(pages)} pages")


# ── Qwen3 NER ───────────────────────────────────────────────────────────

def run_qwen3_ner(pages: List[Dict], ocr_keys: List[str], model_id: Optional[str] = None, ner_dir=None, ner_engine: str = "qwen3") -> None:
    """
    Run Qwen3-8B NER on pages for each OCR output using vLLM.
    ocr_keys: e.g. ["qwenvl_ocr", "deepseek_ocr"]
    Writes 'ner_qwenvl', 'ner_deepseek' keys in-place.
    """
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from transformers import AutoTokenizer

    mid = model_id or "Qwen/Qwen3-8B"
    engine = LLMEngine.from_engine_args(EngineArgs(model=mid, dtype="bfloat16"))
    tokenizer = AutoTokenizer.from_pretrained(mid)

    key_to_ner = {ocr_key: f"ner_{ocr_key.replace('_ocr', '')}" for ocr_key in ocr_keys}
    params = SamplingParams(max_tokens=4096, temperature=0)

    # Build request list: (request_id, prompt_text, page_idx, ocr_key)
    requests = []
    for i, page in enumerate(pages):
        for ocr_key in ocr_keys:
            text = page.get(ocr_key, "")
            if not text or not text.strip():
                page[key_to_ner[ocr_key]] = []
                continue

            if i == 0:
                print(f"\n  [DEBUG] {ocr_key} page 0 OCR text ({len(text)} chars):")
                print(f"  {text[:300]}...")

            messages = [{"role": "user", "content": NER_PROMPT.replace("{text}", text)}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )

            if i == 0:
                print(f"  [DEBUG] Total prompt tokens: {len(tokenizer.encode(prompt))}")

            requests.append((str(len(requests)), prompt, i, ocr_key))

    print(f"  Qwen3 NER: {len(requests)} prompts, generating and saving per-page...")

    # Map request_id -> (page_idx, ocr_key)
    req_map = {req_id: (page_idx, ocr_key) for req_id, _, page_idx, ocr_key in requests}

    # Add all requests upfront
    for req_id, prompt, _, _ in requests:
        engine.add_request(req_id, prompt, params)

    completed = 0
    while engine.has_unfinished_requests():
        for output in engine.step():
            if not output.finished:
                continue
            page_idx, ocr_key = req_map[output.request_id]
            response = output.outputs[0].text
            parsed = parse_ner_response(response)
            pages[page_idx][key_to_ner[ocr_key]] = parsed

            if ner_dir is not None:
                p = pages[page_idx]
                out_path = ner_dir / f"{p['stem']}_{ner_engine}.json"
                with open(out_path, "w") as f:
                    json.dump({"stem": p["stem"], "template_type": p["template_type"],
                               "ocr_engine": ocr_key.replace("_ocr", ""), "ner_engine": ner_engine,
                               "entities": parsed}, f, indent=2)

            completed += 1
            if completed % 50 == 0:
                print(f"  {completed}/{len(requests)} done")

            if page_idx == 0:
                print(f"  [DEBUG] Raw NER response ({len(response)} chars):")
                print(f"  {response[:500]}")
                print(f"  [DEBUG] Parsed {len(parsed)} entities")
                if parsed:
                    print(f"  [DEBUG] First 3: {parsed[:3]}")

    del engine
    import torch
    torch.cuda.empty_cache()
    print(f"Qwen3 NER done — {len(pages)} pages ({len(requests)} prompts)")


# ── VLM Segmentation ────────────────────────────────────────────────────

def run_qwenvl_seg(seg_pages: List[Dict]) -> None:
    """Run Qwen3-VL layout detection. Writes 'qwenvl_regions' key in-place."""
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from PIL import Image

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
    )

    for i, page in enumerate(seg_pages):
        img = Image.open(page["image_path"])
        prompt = SEG_PROMPT.format(width=page["width"], height=page["height"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_input], images=[img], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=8192, do_sample=False)
        response = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        page["qwenvl_regions"] = parse_seg_response(response)
        if (i + 1) % 10 == 0:
            print(f"  Qwen-VL Seg: {i + 1}/{len(seg_pages)}")

    del model, processor
    torch.cuda.empty_cache()
    print(f"Qwen-VL segmentation done — {len(seg_pages)} pages")


def run_deepseek_seg(seg_pages: List[Dict]) -> None:
    """Run DeepSeek-OCR-2 layout detection. Writes 'deepseek_regions' key in-place."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    _patch_llama_compat()

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR-2", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "deepseek-ai/DeepSeek-OCR-2", trust_remote_code=True, use_safetensors=True
    ).eval()

    if torch.cuda.is_available():
        model = model.cuda().to(torch.bfloat16)

    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="deepseek_seg_")

    import io, sys
    for i, page in enumerate(seg_pages):
        prompt = SEG_PROMPT.format(width=page["width"], height=page["height"])
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            model.infer(
                tokenizer,
                prompt=f"<image>\n{prompt}",
                image_file=page["image_path"],
                output_path=tmp_dir,
                base_size=1024, image_size=768, crop_mode=True,
            )
        finally:
            sys.stdout = old_stdout
        page["deepseek_regions"] = parse_seg_response(buf.getvalue().strip())
        if (i + 1) % 10 == 0:
            print(f"  DeepSeek Seg: {i + 1}/{len(seg_pages)}")

    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"DeepSeek segmentation done — {len(seg_pages)} pages")


# ── GLM-OCR ────────────────────────────────────────────────────────────

def run_glm_ocr(pages: List[Dict], model_id: Optional[str] = None) -> None:
    """Run GLM-OCR on pages using vLLM for batched inference. Writes 'glm_ocr' key in-place."""
    from vllm import LLM, SamplingParams
    from PIL import Image

    mid = model_id or "zai-org/GLM-OCR"
    llm = LLM(model=mid, dtype="bfloat16", limit_mm_per_prompt={"image": 1})
    sampling = SamplingParams(temperature=0.0, max_tokens=8192)
    tokenizer = llm.get_tokenizer()

    for i, page in enumerate(pages):
        img = Image.open(page["image_path"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Text Recognition:"},
        ]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        output = llm.generate({
            "prompt": prompt,
            "multi_modal_data": {"image": img},
        }, sampling=sampling)
        page["glm_ocr"] = output[0].outputs[0].text.strip()
        if (i + 1) % 10 == 0:
            print(f"  GLM-OCR: {i + 1}/{len(pages)}")

    del llm
    import torch
    torch.cuda.empty_cache()
    print(f"GLM-OCR done — {len(pages)} pages")


# ── Fine-tuned NER (PEFT checkpoint) ──────────────────────────────────────────

def run_finetuned_ner(
    pages: List[Dict],
    ocr_key: str,
    checkpoint_path,
    ner_engine: str = "finetuned",
    ner_dir=None,
) -> None:
    """Run a fine-tuned NER PEFT checkpoint on pages.

    Uses the same prompt format the model was trained with (NER_TEXT_USER_PROMPT
    from sdd_finetune) — NOT the zero-shot NER_PROMPT used by run_qwen3_ner.
    """
    import torch
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Same prompt used during SFT training
    FINETUNED_SYSTEM = (
        "You are a named entity recognition model. "
        "Given document text, extract all named entities with their types."
    )
    FINETUNED_USER = (
        "Extract all named entities from the following document text.\n\n"
        "Entity types: PERSON, ORG, DATE, ADDRESS, AMOUNT, PHONE, ID\n\n"
        "Respond with a JSON array:\n"
        "```json\n"
        "[\n"
        '  {"type": "PERSON", "text": "John Smith"},\n'
        '  {"type": "ORG", "text": "Acme Corp"}\n'
        "]\n"
        "```\n\n"
        "Document text:\n{text}"
    )

    ck = Path(checkpoint_path)
    adapter_cfg_path = ck / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json in {ck}. "
            "Run sdd-finetune training first, or check the checkpoint path in engines.yaml."
        )

    adapter_cfg = json.loads(adapter_cfg_path.read_text())
    base_model_id = adapter_cfg["base_model_name_or_path"]
    print(f"  Loading fine-tuned checkpoint: {ck}")
    print(f"  Base model: {base_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(ck))
    model.eval()

    ner_out_key = f"ner_{ocr_key.replace('_ocr', '')}"

    for i, page in enumerate(pages):
        text = page.get(ocr_key, "")
        if not text or not text.strip():
            page[ner_out_key] = []
            continue

        messages = [
            {"role": "system", "content": FINETUNED_SYSTEM},
            {"role": "user", "content": FINETUNED_USER.replace("{text}", text)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

        response = tokenizer.decode(
            outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
        parsed = parse_ner_response(response)
        page[ner_out_key] = parsed

        if ner_dir is not None:
            out_path = Path(ner_dir) / f"{page['stem']}_{ner_engine}.json"
            with open(out_path, "w") as f:
                json.dump({
                    "stem": page["stem"], "template_type": page["template_type"],
                    "ocr_engine": ocr_key.replace("_ocr", ""), "ner_engine": ner_engine,
                    "entities": parsed,
                }, f, indent=2)

        if (i + 1) % 10 == 0:
            print(f"  Finetuned NER ({ner_engine}): {i + 1}/{len(pages)}")

    del model, base
    torch.cuda.empty_cache()
    print(f"Finetuned NER ({ner_engine}) done — {len(pages)} pages")
