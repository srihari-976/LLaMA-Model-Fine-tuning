# LLaMA 3.2 3B — Personal AI Assistant

Fine-tune LLaMA 3.2 3B on **4GB RTX 3050** (8-bit QLoRA) to answer anything about [Srihari R](https://github.com/srihari-976).

Trained on **155 Q&A pairs** from GitHub, LinkedIn, Google Scholar (3 IEEE papers), personal website, ORCID, and 42+ projects.

## Quick Start

```bash
pip install -r requirements.txt
huggingface-cli login          # get token from huggingface.co/settings/tokens
python train_qlora.py          # trains on srihari_dataset.json (~20-30 min)
streamlit run app.py           # launch the chatbot UI
```

Or use terminal:

```bash
python inference_qlora.py                              # interactive chat
python inference_qlora.py "What projects have you built?"  # one-off Q&A
```

## What's Inside

| File | What it does |
|------|-------------|
| `srihari_dataset.json` | **155** Q&A pairs — background, projects, research papers, skills, links |
| `train_qlora.py` | QLoRA training: 8-bit, 5 epochs, lr=1.5e-4, max_seq=512, eval + early stopping |
| `train_full.py` | **Whole-model (full) fine-tuning** on a small model that fits 4GB VRAM |
| `tinyLama.py` | Same QLoRA training with OOM fallback + helpful error messages |
| `app.py` | **Streamlit chatbot** — sample questions sidebar, creativity slider, chat history |
| `inference_qlora.py` | CLI interactive or one-shot Q&A (QLoRA **or** full model via `--full`) |
| `config.py` | All shared config (models, quantization, LoRA, training args, seed) |
| `utils.py` | Shared helpers (dataset loading, model loading, tokenization, generation) |
| `ds_config.json` | DeepSpeed CPU-offload config (optional, for larger full-FT models) |
| `requirements.txt` | PyTorch, transformers, peft, bitsandbytes, streamlit, etc. |

## Full Fine-Tuning (whole model)

**Reality check for a 4GB RTX 3050:** full fine-tuning trains *every* weight. Memory needed ≈
4-6 bytes per parameter on top of 2-byte weights. That means:

| Model | Params | Est. VRAM | Fits 4GB? |
|-------|--------|-----------|-----------|
| GPT-2 (`gpt2`) | 124M | ~0.9 GB | ✅ (default) |
| Pythia-160m (`EleutherAI/pythia-160m`) | 162M | ~1.2 GB | ✅ |
| TinyLlama 1.1B | 1.1B | ~8.3 GB | ❌ |
| LLaMA 3.2 3B | 3B | ~22.5 GB | ❌ |

So **LLaMA 3B cannot be fully fine-tuned on 4GB VRAM** (weights alone are 6GB). For true
whole-model fine-tuning, use a small model:

```bash
python train_full.py                                          # full fine-tune gpt2 (default)
FULL_FT_MODEL=EleutherAI/pythia-160m python train_full.py     # slightly bigger / better
```

Optional: leverage your 32GB RAM to full fine-tune larger models via CPU offload
(slow — weeks on CPU-heavy models, not recommended on this i5):

```bash
pip install deepspeed
USE_CPU_OFFLOAD=1 FULL_FT_MODEL=EleutherAI/pythia-410m python train_full.py
```

Chat with the fully fine-tuned model:

```bash
python inference_qlora.py --full ./gpt2-full-ft-out "What projects have you built?"
```

## Hardware Optimized (4GB RTX 3050)

- 8-bit quantization (~3GB for model weights)
- CPU offloading when VRAM fills
- Gradient checkpointing
- paged_adamw_8bit (optimizer on CPU)
- Batch size 1, gradient accumulation 8

## Data Coverage

**Links** — GitHub, LinkedIn (2.4K followers), website, Google Scholar, ORCID, Instagram, Medium  
**Research** — 3 papers: Endpoint Security (IEEE 2025), JobSphere (arXiv), Data Viz (IEEE ASIANCON 2025)  
**Roles** — Co-Lead GDG AI Team @ Presidency University, IEEE member  
**Projects** — WebForge (CrewAI multi-agent), LUMINA (LLaMA 3.1 70B), Data Viz Platform (React+Flask+TF), Spam Classifier (97.67%), AI Chatbot (Flask+MongoDB), ChatNova (Android), SEcureX (blockchain), FACE_RECOGNIZATION (OpenCV), Quantum Computing, HAL internship, Crop Prediction, Traffic Dashboard, and 30+ more  
**Skills** — Python, Java, JavaScript, TypeScript, C#, Solidity, TensorFlow, PyTorch, scikit-learn, OpenCV, React, Flask, Node.js, AWS, GCP, Docker, Git  
**Hackathons** — Pack Hack (24h, quantum-resistant cryptography with Kyber+AES-256)

## Troubleshooting

```bash
# CUDA OOM during QLoRA — reduce sequence length or use TinyLlama
MAX_SEQ_LENGTH=256 python train_qlora.py
BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0" python train_qlora.py

# Full fine-tuning OOM — pick a model that fits (see table above)
FULL_FT_MODEL=EleutherAI/pythia-160m python train_full.py
```

## Links

- 🐙 GitHub: https://github.com/srihari-976
- 💼 LinkedIn: https://linkedin.com/in/srihari-r-614714252
- 🌐 Website: https://www.sriharir.tech/
- 📄 Google Scholar: https://scholar.google.com/citations?user=smM0D5UAAAAJ
- 🆔 ORCID: https://orcid.org/0009-0007-7765-4474
