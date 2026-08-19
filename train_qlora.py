import os
import sys

# Ensure project root is on path for config/utils imports
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from datasets import load_dataset
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
    EarlyStoppingCallback,
    compute_warmup_steps,
    format_example,
    load_base_model,
    load_train_eval_datasets,
    set_seed,
    tokenize_for_training,
    validate_dataset_schema,
)

set_seed(SEED)


MODEL_NAME = os.environ.get("BASE_MODEL", BASE_MODEL)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", OUTPUT_DIR)
DATASET = os.environ.get("DATASET", "")
DATASET_FILE = os.environ.get("DATASET_FILE", DATASET_FILE)


def load_datasets():
    """Return (train_dataset, eval_dataset) formatted as 'text'."""
    if DATASET:
        raw = load_dataset(DATASET, split="train")
        validate_dataset_schema(raw)
        formatted = raw.map(format_example, remove_columns=raw.column_names)
        split = formatted.train_test_split(test_size=0.1, seed=SEED)
        return split["train"], split["test"]
    return load_train_eval_datasets(DATASET_FILE)


def main():
    # ── Load model (8-bit quantized) + tokenizer ──────────────────────────
    model, tokenizer = load_base_model(model_name=MODEL_NAME)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset, eval_dataset = load_datasets()
    tokenized_train = tokenize_for_training(train_dataset, tokenizer, MAX_SEQ_LENGTH)
    tokenized_eval = tokenize_for_training(eval_dataset, tokenizer, MAX_SEQ_LENGTH)

    # ── Training arguments (base from config + eval settings) ─────────────
    training_args = TrainingArguments(
        **{
            **TRAINING_ARGS.to_dict(),
            "output_dir": OUTPUT_DIR,
            "eval_strategy": "steps",
            "eval_steps": 50,
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
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. Model saved to {OUTPUT_DIR}")
    print(
        f"Dataset size: {len(tokenized_train)} training + {len(tokenized_eval)} eval examples"
    )


if __name__ == "__main__":
    main()
