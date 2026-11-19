#!/usr/bin/env python3
"""
OmniDocBench driver for local VLM-based PDF extractors (Clariden / GH200).

Uses HuggingFace transformers directly (no vLLM): the NGC PyTorch container
ships a working torch 2.6.0a0+nv24.12 build optimized for Grace-Hopper, and
the pip-installable vLLM stack proved brittle on aarch64+cu124.

Two phases driven by the SLURM wrapper:

  1. Extraction phase: for each page-image, prompt the VLM to produce
     markdown; predictions saved to <output_dir>/preds/<page_id>.md.
  2. Scoring phase (--score-only): edit-distance + fuzz vs ground-truth
     markdown from the dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("omnidoc")


EXTRACTION_PROMPT = (
    "You are a document parser. Convert the page image to clean Markdown.\n"
    "Rules:\n"
    "- Preserve reading order (multi-column aware).\n"
    "- Render tables as GitHub-flavored Markdown tables.\n"
    "- Render math as LaTeX inside $...$ or $$...$$.\n"
    "- Do NOT add commentary, headers, or explanations — output Markdown only."
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def iter_pages(dataset_dir: Path) -> Iterable[tuple[str, Path, str]]:
    """Yield (page_id, image_path, gold_markdown) for every annotated page."""
    manifest = dataset_dir / "OmniDocBench.json"
    if not manifest.exists():
        log.error(f"Manifest not found: {manifest}")
        sys.exit(2)

    data = json.loads(manifest.read_text())
    for entry in data:
        page_id = entry.get("page_info", {}).get("image_path") or entry.get(
            "image_path"
        )
        if not page_id:
            continue
        img = dataset_dir / "images" / Path(page_id).name
        if not img.exists():
            log.warning(f"missing image for {page_id}")
            continue
        gold = entry.get("markdown") or entry.get("gt", {}).get("markdown") or ""
        yield Path(page_id).stem, img, gold


# ---------------------------------------------------------------------------
# Extraction via HuggingFace transformers
# ---------------------------------------------------------------------------
def load_model(model_id: str):
    """Load a VLM via AutoModelForImageTextToText + AutoProcessor.

    Works for Qwen2.5-VL, Llama-3.2-Vision, Nemotron-VL, InternVL, etc.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    log.info(f"Loading {model_id} (dtype=bfloat16, device=cuda)…")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def generate(
    model, processor, image, instruction: str, max_new_tokens: int = 4096
) -> str:
    """Single-image VLM generation via the model's chat template."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    # Strip the prompt prefix so we only return generated tokens.
    gen_ids = out_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]


def run_extraction(args: argparse.Namespace) -> None:
    from PIL import Image

    out_dir = args.output_dir / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model_id)

    pairs = list(iter_pages(args.dataset_dir))
    if args.limit:
        pairs = pairs[: args.limit]
    log.info(f"Extracting on {len(pairs)} pages")

    timings = []
    for i, (pid, img_path, _gold) in enumerate(pairs, 1):
        pred_path = out_dir / f"{pid}.md"
        if pred_path.exists() and not args.overwrite:
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail(_pixel_budget_box(args.max_pixels))
            t0 = time.perf_counter()
            text = generate(model, processor, img, EXTRACTION_PROMPT)
            dt = time.perf_counter() - t0
            pred_path.write_text(text)
            timings.append(dt)
            if i % 5 == 0 or i == len(pairs):
                log.info(
                    f"  [{i}/{len(pairs)}] last={dt:.1f}s  mean={statistics.mean(timings):.1f}s"
                )
        except Exception as e:
            log.error(f"  page {pid} failed: {e}")
            (out_dir / f"{pid}.err").write_text(repr(e))

    (args.output_dir / "extraction_meta.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "tag": args.tag,
                "n_pages": len(pairs),
                "mean_latency_s": statistics.mean(timings) if timings else None,
                "total_s": sum(timings),
            },
            indent=2,
        )
    )
    log.info("Extraction done.")


def _pixel_budget_box(max_pixels: int) -> tuple[int, int]:
    side = int(max_pixels**0.5)
    return (side, side)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def run_scoring(args: argparse.Namespace) -> None:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein

    preds_dir = args.output_dir / "preds"
    per_doc = []
    for pid, _img, gold in iter_pages(args.dataset_dir):
        pred_file = preds_dir / f"{pid}.md"
        if not pred_file.exists():
            continue
        hyp = pred_file.read_text()
        ref = gold or ""
        if not ref.strip():
            continue
        per_doc.append(
            {
                "page_id": pid,
                "edit_distance_norm": Levenshtein.distance(ref, hyp) / max(len(ref), 1),
                "fuzz_ratio": fuzz.ratio(ref, hyp) / 100.0,
                "ref_chars": len(ref),
                "hyp_chars": len(hyp),
            }
        )
    if not per_doc:
        log.error("No scorable pages — did extraction run?")
        sys.exit(3)

    agg = {
        "n_pages": len(per_doc),
        "edit_distance_mean": statistics.mean(d["edit_distance_norm"] for d in per_doc),
        "edit_distance_median": statistics.median(
            d["edit_distance_norm"] for d in per_doc
        ),
        "fuzz_ratio_mean": statistics.mean(d["fuzz_ratio"] for d in per_doc),
    }
    out = args.output_dir / f"scores_{args.tag}.json"
    out.write_text(
        json.dumps({"tag": args.tag, "aggregate": agg, "per_doc": per_doc}, indent=2)
    )
    log.info(f"Wrote {out}")
    log.info(f"Aggregate: {agg}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--max-pixels", type=int, default=1_600_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        run_scoring(args)
    else:
        if not args.model_id:
            ap.error("--model-id is required for extraction phase")
        run_extraction(args)


if __name__ == "__main__":
    main()
