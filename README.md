# LLaMA 3.2 3B — Personal AI Assistant

Fine-tune LLaMA 3.2 3B on **4GB RTX 3050** (8-bit QLoRA) to answer anything about [Srihari R](https://github.com/srihari-976).

Trained on **130 Q&A pairs** from GitHub, LinkedIn, Google Scholar (3 IEEE papers), personal website, ORCID, and 42+ projects.

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
| `srihari_dataset.json` | **130** Q&A pairs — background, 42 projects, 3 research papers, skills, links |
| `train_qlora.py` | Training script: 8-bit QLoRA, 5 epochs, lr=1.5e-4, max_seq=512 |
| `tinyLama.py` | Same training with OOM fallback + helpful error messages |
| `app.py` | **Streamlit chatbot** — sample questions sidebar, creativity slider, chat history |
| `inference_qlora.py` | CLI interactive or one-shot Q&A |
| `requirements.txt` | PyTorch, transformers, peft, trl, bitsandbytes, streamlit, etc. |

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
# CUDA OOM — reduce sequence length or use TinyLlama
max_seq_length=256 python train_qlora.py
BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0" python train_qlora.py
```

## Links

- 🐙 GitHub: https://github.com/srihari-976
- 💼 LinkedIn: https://linkedin.com/in/srihari-r-614714252
- 🌐 Website: https://www.sriharir.tech/
- 📄 Google Scholar: https://scholar.google.com/citations?user=smM0D5UAAAAJ
- 🆔 ORCID: https://orcid.org/0009-0007-7765-4474
