import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER_DIR = "./llama3-qlora-out"


def load_model(base_model=BASE_MODEL, adapter_dir=ADAPTER_DIR):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_threshold=6.0,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("### Response:\n")[-1].strip()


if __name__ == "__main__":
    import sys
    print("Loading model...")
    try:
        model, tokenizer = load_model()
        print("Ready! Type 'quit' to exit.\n")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrain first: python train_qlora.py")
        sys.exit(1)

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(f"Q: {prompt}")
        print(f"A: {generate(model, tokenizer, prompt)}")
    else:
        while True:
            try:
                user = input("You: ").strip()
                if user.lower() in ("quit", "exit", "q"):
                    print("Bye!")
                    break
                if not user:
                    continue
                print(f"AI: {generate(model, tokenizer, user)}\n")
            except KeyboardInterrupt:
                print("\nBye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")
