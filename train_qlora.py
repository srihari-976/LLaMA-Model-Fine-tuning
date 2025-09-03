import os, torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, AutoConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from accelerate import init_empty_weights, infer_auto_device_map

# --- Configuration ---
MODEL_NAME = os.environ.get("BASE_MODEL", "meta-llama/Llama-2-7b-hf")
DATASET = os.environ.get("DATASET", "yahma/alpaca-cleaned")

# --- 4-bit Quantization (QLoRA) Config ---
# This configuration enables loading the model in 4-bit precision.
# FIX: llm_int8_enable_fp32_cpu_offload=True is re-added to allow CPU offloading.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Set padding side

# --- Model Loading ---
# Pre-calculate the device map to avoid the "meta tensor" error.
# This is a more robust way to handle mixed GPU/CPU setups.

# 1. Get the model configuration
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 2. Create an empty model on the meta device (no memory used)
with init_empty_weights():
    model_empty = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model_empty.tie_weights()

# 3. Define max memory usage and infer the device map
max_memory = {0: "3.5GB", "cpu": "28GB"}
device_map = infer_auto_device_map(
    model_empty,
    max_memory=max_memory,
    no_split_module_classes=["LlamaDecoderLayer"], # Prevents splitting layers
    dtype="float16"
)

# 4. Load the actual model with the pre-calculated device map
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map=device_map, # Use the calculated map
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Prepare the model for k-bit training. This function makes the model ready for training
# by correctly setting up the input embeddings and other components to handle gradients.
model = prepare_model_for_kbit_training(model)

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
# SFTConfig is deprecated. We now use TrainingArguments directly.
training_args = TrainingArguments(
    output_dir="qlora-out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4, # Reduced to save memory
    gradient_checkpointing=True, # Enable gradient checkpointing here
    num_train_epochs=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to=[],
    ddp_find_unused_parameters=False,
    max_grad_norm=0.3, # Added for training stability
    dataloader_pin_memory=False # Can help with memory issues
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
subset_size = min(200, len(train_dataset))
train_subset = train_dataset.select(range(subset_size))

# --- Trainer Initialization ---
# The trainer will use its default settings for sequence length and packing.
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
)

# --- Training ---
trainer.train()

# --- Save Artifacts ---
trainer.save_model("qlora-out")
tokenizer.save_pretrained("qlora-out")
print("Done. Adapters and tokenizer saved in ./qlora-out")
