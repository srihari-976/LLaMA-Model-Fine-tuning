import json
import math
import os
import random
import sys
from typing import Any

import numpy as np
import torch
import transformers
from datasets import Dataset, load_dataset
from transformers import BitsAndBytesConfig, TrainerCallback

# Ensure project root is on path so config is importable from any CWD
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import (
    BASE_MODEL,
    BNB_CONFIG,
    DATASET_FILE,
    GENERATION_ARGS,
    OUTPUT_DIR,
    SEED,
)

# ── Reproducibility ────────────────────────────────────────────────────────


def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility across numpy, python, torch, and transformers."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


# ── Training callbacks ─────────────────────────────────────────────────────


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when eval loss does not improve for `patience` evaluations."""

    def __init__(self, patience: int = 2):
        self.patience = patience
        self.best_eval_loss = float("inf")
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(
                    f"\nEarly stopping triggered after {state.global_step} steps "
                    f"(eval_loss={eval_loss:.4f})"
                )
                control.should_training_stop = True


# ── Prompt formatting ─────────────────────────────────────────────────────


def format_example(example: dict) -> dict:
    """Format a dataset example into the Alpaca-style prompt string."""
    instr = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    if inp:
        prompt = (
            f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        )
    else:
        prompt = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
    return {"text": prompt}


def validate_dataset_schema(dataset) -> None:
    """Ensure dataset has required fields; raise if missing."""
    if "instruction" not in dataset.column_names:
        raise ValueError("Dataset must contain 'instruction' column")
    if "output" not in dataset.column_names:
        raise ValueError("Dataset must contain 'output' column")


# ── Training helpers ───────────────────────────────────────────────────────


def compute_warmup_steps(
    num_train_examples: int,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 5,
    warmup_ratio: float = 0.05,
) -> int:
    """Convert a warmup ratio into an absolute step count for TrainingArguments.

    `warmup_ratio` was removed from `TrainingArguments` in transformers 5.15,
    so warmup must be given as `warmup_steps`.
    """
    steps_per_epoch = math.ceil(
        num_train_examples / (batch_size * gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * num_epochs
    return max(1, int(warmup_ratio * total_steps))


# ── Model loading ──────────────────────────────────────────────────────────


def load_base_model(
    model_name: str = BASE_MODEL,
    bnb_config: BitsAndBytesConfig | None = None,
    torch_dtype: torch.dtype = torch.float16,
) -> tuple:
    """Load a base model with 8-bit quantization and device map auto."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bnb_cfg = bnb_config or BNB_CONFIG
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch_dtype,
    )
    return model, tokenizer


def load_with_adapter(
    base_model,
    adapter_dir: str = OUTPUT_DIR,
) -> Any:
    """PEFT model with loaded adapter weights."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model


def load_full_model(
    model_dir: str,
    torch_dtype: torch.dtype = torch.float16,
) -> tuple:
    """Load a fully fine-tuned (whole-model) checkpoint for inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch_dtype)
    model.eval()
    return model, tokenizer


# ── Generation ─────────────────────────────────────────────────────────────


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = GENERATION_ARGS["max_new_tokens"],
    temperature: float = GENERATION_ARGS["temperature"],
    top_p: float = GENERATION_ARGS["top_p"],
    repetition_penalty: float = GENERATION_ARGS["repetition_penalty"],
    max_new_tokens_max: int = 512,
) -> str:
    """Generate text from a prompt using the fine-tuned model."""
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Clamp max_new_tokens to a safe range (>=1 and <= configured cap)
    effective_max = max(1, min(max_new_tokens, max_new_tokens_max))

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=effective_max,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("### Response:\n")[-1].strip()


# ── Dataset helpers ────────────────────────────────────────────────────────


def load_raw_dataset(dataset_path: str) -> Dataset:
    """Load a raw (instruction/input/output) dataset from JSON.

    Supports both a flat array of records and a wrapped object such as
    ``{"records": [...]}``.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "records" in data:
        field = "records"
    else:
        field = None

    kwargs: dict = {"data_files": dataset_path, "split": "train"}
    if field is not None:
        kwargs["field"] = field
    return load_dataset("json", **kwargs)


def load_train_dataset(dataset_path: str = DATASET_FILE) -> Dataset:
    """Load and format a single training dataset (no split)."""
    if dataset_path and os.path.exists(dataset_path):
        raw = load_raw_dataset(dataset_path)
    else:
        raw = load_dataset("yahma/alpaca-cleaned", split="train")

    validate_dataset_schema(raw)
    return raw.map(format_example, remove_columns=raw.column_names)


def load_train_eval_datasets(
    dataset_path: str = DATASET_FILE,
    test_size: float = 0.1,
    seed: int = SEED,
) -> tuple[Dataset, Dataset]:
    """Load, format, and split a dataset into (train, eval) reproducibly."""
    formatted = load_train_dataset(dataset_path)
    split = formatted.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def tokenize_for_training(
    dataset: Dataset,
    tokenizer,
    max_seq_length: int = 512,
) -> Dataset:
    """Tokenize a formatted ('text' column) dataset for causal LM training."""

    def tokenize(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
        )

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
