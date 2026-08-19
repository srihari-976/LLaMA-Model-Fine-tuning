import os
import sys

# Ensure project root is on path for config/utils imports
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import (
    BASE_MODEL,
    FULL_FT_OUTPUT_DIR,
    OUTPUT_DIR,
    SEED,
)
from utils import (
    generate_text,
    load_base_model,
    load_full_model,
    load_with_adapter,
    set_seed,
)

set_seed(SEED)


def load_model(base_model=None, adapter_dir=None):
    base_model = base_model or BASE_MODEL
    adapter_dir = adapter_dir or OUTPUT_DIR
    model, tokenizer = load_base_model(model_name=base_model)
    model = load_with_adapter(model, adapter_dir=adapter_dir)
    return model, tokenizer


def load_model_full(model_dir=None):
    model_dir = model_dir or FULL_FT_OUTPUT_DIR
    return load_full_model(model_dir)


def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    return generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QLoRA / full fine-tune inference")
    parser.add_argument("prompt", nargs="?", help="Prompt to query the model")
    parser.add_argument("--model", default=OUTPUT_DIR, help="Adapter directory")
    parser.add_argument("--base", default=BASE_MODEL, help="Base model name")
    parser.add_argument(
        "--full",
        metavar="DIR",
        default=None,
        help="Load a fully fine-tuned (whole-model) checkpoint from DIR instead of base+adapter",
    )
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--tokens", type=int, default=256, help="Max new tokens")
    args = parser.parse_args()

    print("Loading model...")
    try:
        if args.full:
            model, tokenizer = load_model_full(args.full)
            print(f"Loaded full model from {args.full}\n")
        else:
            model, tokenizer = load_model(base_model=args.base, adapter_dir=args.model)
            print("Ready!\n")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrain first: python train_qlora.py  (or  python train_full.py)")
        sys.exit(1)

    if args.prompt:
        print(f"Q: {args.prompt}")
        print(
            f"A: {generate(model, tokenizer, args.prompt, max_new_tokens=args.tokens, temperature=args.temp)}"
        )
    else:
        while True:
            try:
                user = input("You: ").strip()
                if user.lower() in ("quit", "exit", "q"):
                    print("Bye!")
                    break
                if not user:
                    continue
                print(
                    f"AI: {generate(model, tokenizer, user, max_new_tokens=args.tokens, temperature=args.temp)}\n"
                )
            except KeyboardInterrupt:
                print("\nBye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")
