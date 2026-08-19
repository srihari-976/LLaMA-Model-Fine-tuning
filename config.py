import os

from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainingArguments

# ── Model paths ────────────────────────────────────────────────────────────
BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-3B")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./llama3-qlora-out")
DATASET_FILE = os.environ.get("DATASET_FILE", "srihari_dataset.json")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "512"))

# ── Full fine-tuning (whole-model) ─────────────────────────────────────────
# Whole-model fine-tuning needs ~4 bytes/param on top of 2-byte weights.
# A 4GB RTX 3050 can only fully fine-tune models up to ~150-250M params.
# Defaults to gpt2 (124M) which fits comfortably. Override with FULL_FT_MODEL.
FULL_FT_MODEL = os.environ.get("FULL_FT_MODEL", "gpt2")
FULL_FT_OUTPUT_DIR = os.environ.get("FULL_FT_OUTPUT_DIR", "./gpt2-full-ft-out")

# ── Quantization ───────────────────────────────────────────────────────────
BNB_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_threshold=6.0,
)

# ── LoRA ───────────────────────────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# ── Training (QLoRA) ───────────────────────────────────────────────────────
# NOTE: warmup is configured per-script as `warmup_steps` (computed from the
# dataset size) because `warmup_ratio` was removed in newer transformers.
TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine",
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to="none",
    max_grad_norm=0.3,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    load_best_model_at_end=False,
)

# ── Training (Full / whole-model) ──────────────────────────────────────────
FULL_FT_TRAINING_ARGS = TrainingArguments(
    output_dir=FULL_FT_OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to="none",
    max_grad_norm=0.3,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    load_best_model_at_end=False,
)

# ── Default generation ────────────────────────────────────────────────────
GENERATION_ARGS = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42
