import os
import sys

# Ensure project root is on path for config/utils imports
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import (
    BASE_MODEL,
    DATASET_FILE,
    LORA_CONFIG,
    MAX_SEQ_LENGTH,
    OUTPUT_DIR,
    SEED,
    TRAINING_ARGS,
)
from utils import (
    compute_warmup_steps,
    load_base_model,
    load_train_dataset,
    set_seed,
    tokenize_for_training,
)

set_seed(SEED)


MODEL_NAME = os.environ.get("BASE_MODEL", BASE_MODEL)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", OUTPUT_DIR)
DATASET_FILE = os.environ.get("DATASET_FILE", DATASET_FILE)


def main():
    # ── Load model (8-bit quantized) + tokenizer ──────────────────────────
    model, tokenizer = load_base_model(model_name=MODEL_NAME)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # ── Dataset (full dataset, no eval split) ─────────────────────────────
    train_dataset = load_train_dataset(DATASET_FILE)
    tokenized_train = tokenize_for_training(train_dataset, tokenizer, MAX_SEQ_LENGTH)

    # ── Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        **{
            **TRAINING_ARGS.to_dict(),
            "output_dir": OUTPUT_DIR,
            "warmup_steps": compute_warmup_steps(
                len(tokenized_train),
                batch_size=1,
                gradient_accumulation_steps=8,
                num_epochs=5,
            ),
        }
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ── Train with helpful OOM guidance ───────────────────────────────────
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\nCUDA OOM on 4GB GPU. Try:")
            print("  - Close all other GPU apps (Chrome, etc.)")
            print("  - Set MAX_SEQ_LENGTH=256 in this script")
            print("  - Or use a smaller model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        raise

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. Model saved to {OUTPUT_DIR}")
    print(f"Dataset size: {len(tokenized_train)} examples")


if __name__ == "__main__":
    main()
