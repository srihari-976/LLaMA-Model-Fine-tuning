import os
import sys

# Ensure project root is on path for config/utils imports
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import (
    DATASET_FILE,
    FULL_FT_MODEL,
    FULL_FT_OUTPUT_DIR,
    FULL_FT_TRAINING_ARGS,
    MAX_SEQ_LENGTH,
    SEED,
)
from utils import (
    EarlyStoppingCallback,
    compute_warmup_steps,
    load_train_eval_datasets,
    set_seed,
    tokenize_for_training,
)

set_seed(SEED)


MODEL_NAME = os.environ.get("FULL_FT_MODEL", FULL_FT_MODEL)
OUTPUT_DIR = os.environ.get("FULL_FT_OUTPUT_DIR", FULL_FT_OUTPUT_DIR)
USE_CPU_OFFLOAD = os.environ.get("USE_CPU_OFFLOAD", "0") == "1"
DS_CONFIG = os.path.join(_project_root, "ds_config.json")


def estimate_vram(num_params: int) -> float:
    """Rough GPU VRAM estimate (GB) for full fp16 fine-tuning with 8-bit paged Adam.

    weights (fp16) + gradients (fp16) + optimizer (8-bit paged) + activations.
    """
    weights = num_params * 2
    grads = num_params * 2
    optimizer = num_params * 2
    activations = num_params * 1.5
    return (weights + grads + optimizer + activations) / 1e9


def check_fit(num_params: int) -> None:
    """Warn early if the selected model is unlikely to fit a 4GB GPU."""
    vram_gb = estimate_vram(num_params)
    print(
        f"Model size: {num_params / 1e6:.0f}M params | estimated VRAM: ~{vram_gb:.1f} GB"
    )
    if vram_gb > 4.0 and not USE_CPU_OFFLOAD:
        print(
            f"\nWARNING: {MODEL_NAME} is estimated to need ~{vram_gb:.1f} GB VRAM "
            "for full fine-tuning. A 4GB RTX 3050 will likely run out of memory.\n"
            "Options:\n"
            "  1. Use a smaller model, e.g.: FULL_FT_MODEL=EleutherAI/pythia-160m python train_full.py\n"
            "  2. Use CPU offload (needs DeepSpeed):  USE_CPU_OFFLOAD=1 python train_full.py\n"
            "  3. Fall back to QLoRA (recommended for 3B): python train_qlora.py\n"
        )


def main():
    if not torch.cuda.is_available():
        print(
            "CUDA not detected. Full fine-tuning will run on CPU, which is very slow.\n"
            "If you have an NVIDIA GPU, install CUDA + PyTorch with CUDA support.\n"
        )

    # ── Load FULL model (no quantization, no LoRA) ────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    num_params = sum(p.numel() for p in model.parameters())
    check_fit(num_params)

    if torch.cuda.is_available():
        model = model.to("cuda")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset, eval_dataset = load_train_eval_datasets(DATASET_FILE)
    tokenized_train = tokenize_for_training(train_dataset, tokenizer, MAX_SEQ_LENGTH)
    tokenized_eval = tokenize_for_training(eval_dataset, tokenizer, MAX_SEQ_LENGTH)

    # ── Training arguments ────────────────────────────────────────────────
    args_dict = FULL_FT_TRAINING_ARGS.to_dict()
    args_dict.update(
        {
            "output_dir": OUTPUT_DIR,
            "eval_strategy": "steps",
            "eval_steps": 50,
            "optim": "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            "warmup_steps": compute_warmup_steps(
                len(tokenized_train),
                batch_size=1,
                gradient_accumulation_steps=8,
                num_epochs=5,
            ),
        }
    )

    if USE_CPU_OFFLOAD:
        try:
            import deepspeed  # noqa: F401

            args_dict["deepspeed"] = DS_CONFIG
            print(f"Using DeepSpeed CPU offload config: {DS_CONFIG}")
        except ImportError:
            print("USE_CPU_OFFLOAD=1 requires DeepSpeed:\n  pip install deepspeed")

    training_args = TrainingArguments(**args_dict)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(patience=2)],
    )

    # ── Train (entire model, all weights updated) ─────────────────────────
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. FULL model saved to {OUTPUT_DIR}")
    print(
        f"Trainable params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M (100%)"
    )


if __name__ == "__main__":
    main()
