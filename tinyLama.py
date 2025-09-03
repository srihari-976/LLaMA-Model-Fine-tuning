import os, torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, AutoConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from accelerate import init_empty_weights, infer_auto_device_map

# --- Configuration ---
MODEL_NAME = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # Using a smaller model
DATASET = os.environ.get("DATASET", "yahma/alpaca-cleaned")

# --- 4-bit Quantization (QLoRA) Config ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Model Loading ---
# For 4GB VRAM, we need to use a smaller model and carefully manage memory
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

with init_empty_weights():
    model_empty = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model_empty.tie_weights()

# Conservative memory allocation for 4GB GPU
max_memory = {0: "3GB", "cpu": "24GB"}  # Leave 1GB GPU headroom
device_map = infer_auto_device_map(
    model_empty,
    max_memory=max_memory,
    no_split_module_classes=["LlamaDecoderLayer"],
    dtype="float16"
)

# Load the actual model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# --- LoRA (Low-Rank Adaptation) Config ---
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, peft_config)

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="qlora-out",
    per_device_train_batch_size=1,  # Keep batch size small
    gradient_accumulation_steps=8,  # Reduced from 16 to save memory
    num_train_epochs=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True,
    gradient_checkpointing=True,  # Essential for memory savings
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to=[],
    ddp_find_unused_parameters=False,
    max_grad_norm=0.3,
    dataloader_pin_memory=False,  # Can help with memory issues
)

# --- Dataset Formatting ---
def format_example(example):
    instr = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    if inp:
        prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    else:
        prompt = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
    return {"text": prompt}

# --- Data Loading and Preparation ---
raw_dataset = load_dataset(DATASET, split="train")
train_dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

# Use a smaller subset of data to fit in memory
subset_size = min(500, len(train_dataset))  # Reduced from 1000 to 500
train_subset = train_dataset.select(range(subset_size))

# --- Trainer Initialization ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    dataset_text_field="text",
    max_seq_length=256,  # Reduced sequence length to save memory
    tokenizer=tokenizer,
)

# --- Training ---
try:
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("Out of memory error. Try reducing batch size, sequence length, or dataset size.")
        raise e

# --- Save Artifacts ---
trainer.save_model("qlora-out")
tokenizer.save_pretrained("qlora-out")
print("Done. Adapters and tokenizer saved in ./qlora-out")