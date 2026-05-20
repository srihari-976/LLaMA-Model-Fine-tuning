import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_NAME = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-3B")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./llama3-qlora-out")
DATASET_FILE = os.environ.get("DATASET_FILE", "srihari_dataset.json")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_threshold=6.0,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def format_example(example):
    instr = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    if inp:
        prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    else:
        prompt = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
    return {"text": prompt}


if os.path.exists(DATASET_FILE):
    raw_dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
else:
    print(f"{DATASET_FILE} not found. Using Alpaca dataset.")
    raw_dataset = load_dataset("yahma/alpaca-cleaned", split="train")

train_dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
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
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

try:
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\nCUDA OOM on 4GB GPU. Try:")
        print("  - Close all other GPU apps (Chrome, etc.)")
        print("  - Set max_seq_length=256 in this script")
        print("  - Or use a smaller model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    raise

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Model saved to {OUTPUT_DIR}")
print(f"Dataset size: {len(train_dataset)} examples")
