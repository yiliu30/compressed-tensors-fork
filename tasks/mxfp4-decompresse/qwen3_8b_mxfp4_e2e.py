# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


DEFAULT_MODEL_PATH = Path("/mnt/disk1/yiliu7/Qwen/Qwen3-8B")
DEFAULT_OUTPUT_PATH = Path("/mnt/disk1/yiliu7/Qwen/Qwen3-8B-MXFP4-e2e")
DEFAULT_PROMPT = (
    "Write a short paragraph explaining what model quantization is "
    "and why it helps inference."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify MXFP4 end-to-end save/load/generate on Qwen3-8B."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the source Qwen3-8B model.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the MXFP4-compressed model will be saved.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for baseline and compressed generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep an existing output directory instead of deleting it first.",
    )
    return parser.parse_args()


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()

    if args.output_path.exists() and not args.keep_output:
        shutil.rmtree(args.output_path)

    print(f"Loading base model from {args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("Running baseline generation", flush=True)
    baseline = generate_text(
        model, tokenizer, args.prompt, args.max_new_tokens
    )
    print("BASELINE OUTPUT:", flush=True)
    print(baseline, flush=True)

    print("Applying MXFP4 quantization", flush=True)
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="MXFP4",
        ignore=["lm_head"],
    )
    oneshot(model=model, recipe=recipe)

    print(f"Saving compressed model to {args.output_path}", flush=True)
    model.save_pretrained(args.output_path, save_compressed=True)
    tokenizer.save_pretrained(args.output_path)

    print("Reloading compressed model", flush=True)
    reloaded = AutoModelForCausalLM.from_pretrained(
        args.output_path,
        dtype="auto",
        device_map="auto",
    )
    dispatch_model(reloaded)

    print("Running compressed generation", flush=True)
    compressed = generate_text(
        reloaded, tokenizer, args.prompt, args.max_new_tokens
    )
    print("COMPRESSED OUTPUT:", flush=True)
    print(compressed, flush=True)


if __name__ == "__main__":
    main()
