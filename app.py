import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER_DIR = "./llama3-qlora-out"


@st.cache_resource(show_spinner=False)
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_threshold=6.0,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
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


st.set_page_config(
    page_title="Srihari R - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main > div { padding-bottom: 2rem; }
    .user-msg { background-color: #1e3a5f; padding: 1rem; border-radius: 12px; margin: 0.5rem 0; }
    .assistant-msg { background-color: #1a1d24; padding: 1rem; border-radius: 12px; margin: 0.5rem 0; }
    .sample-btn { margin: 0.2rem; }
    h1, h2, h3 { color: #f0f2f6; }
    .sidebar-text { font-size: 0.9rem; color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🤖 Srihari R AI")
st.sidebar.markdown("---")

model_ready = False
try:
    model, tokenizer = load_model()
    model_ready = True
    st.sidebar.success("✅ Model loaded")
except Exception:
    st.sidebar.error("❌ Model not trained yet")
    st.sidebar.markdown("Run this first:\n```bash\npython train_qlora.py\n```")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
temperature = st.sidebar.slider("Creativity", 0.1, 1.5, 0.7, 0.1,
                                help="Lower = precise, Higher = creative")
max_tokens = st.sidebar.slider("Max response length", 64, 512, 256, 32)

if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Try asking")
samples = [
    "Who are you?",
    "What projects have you built?",
    "What research have you published?",
    "What is your LinkedIn?",
    "Tell me about WebForge",
    "What skills do you have?",
    "What is your GitHub?",
    "What hackathons have you done?",
    "Tell me about your education",
    "What is GDG?",
]
for s in samples:
    if st.sidebar.button(s, use_container_width=True, key=s):
        st.session_state.messages.append({"role": "user", "content": s})
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p class="sidebar-text">Fine-tuned on 130+ Q&A pairs from GitHub, LinkedIn, Google Scholar, and personal website.</p>',
    unsafe_allow_html=True
)
st.sidebar.markdown(
    '<p class="sidebar-text"><a href="https://github.com/srihari-976" target="_blank">🐙 GitHub</a> • <a href="https://linkedin.com/in/srihari-r-614714252" target="_blank">💼 LinkedIn</a> • <a href="https://www.sriharir.tech/" target="_blank">🌐 Website</a></p>',
    unsafe_allow_html=True
)

st.title("🤖 Ask Me Anything About Srihari")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about Srihari...", disabled=not model_ready):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = generate(model, tokenizer, prompt, max_tokens, temperature)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except torch.cuda.OutOfMemoryError:
                st.error("GPU ran out of memory. Try restarting or use a shorter prompt.")
            except Exception as e:
                st.error(f"Error generating response: {e}")

if not model_ready and not st.session_state.messages:
    st.info("👋 Welcome! The model isn't trained yet. Run `python train_qlora.py` first, then refresh this page.")
    st.code("pip install -r requirements.txt\nhuggingface-cli login\npython train_qlora.py\nstreamlit run app.py")

elif not st.session_state.messages:
    st.markdown("""
    Hi! I'm an AI assistant trained on **Srihari R's** public profile.  
    Ask me about his background, projects, research, skills, or links.
    
    **Quick links:**  
    🐙 [GitHub](https://github.com/srihari-976)  
    💼 [LinkedIn](https://linkedin.com/in/srihari-r-614714252)  
    🌐 [Website](https://www.sriharir.tech/)  
    📄 [Google Scholar](https://scholar.google.com/citations?user=smM0D5UAAAAJ)
    """)
