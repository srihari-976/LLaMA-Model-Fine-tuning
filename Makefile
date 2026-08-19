.PHONY: help train train-full infer app clean test

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-+]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "} {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

train:  ## Train the QLoRA model on srihari_dataset.json
	python train_qlora.py

train-full:  ## Train the ENTIRE model (full fine-tuning, small model fits 4GB)
	python train_full.py

infer:  ## Run CLI inference (interactive mode)
	python inference_qlora.py

infer-one:  ## Run CLI inference with a prompt
	python inference_qlora.py " $(PROMPT)"

app:  ## Launch Streamlit chatbot UI
	streamlit run app.py

clean:  ## Remove model output directory
	rm -rf ./llama3-qlora-out

install:  ## Install dependencies
	pip install -r requirements.txt

test:  ## Quick import test to verify modules load
	python -c "from config import *; from utils import *; set_seed(); print('All modules loaded OK')"