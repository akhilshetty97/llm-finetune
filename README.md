# Fine-Tuning a Domain-Specific LLM for Radiology Reports with LoRA

This repository contains code for fine-tuning a large language model (LLM), specifically Microsoft's **BioGPT**, to generate radiology reports based on diagnosis confidence scores. We leverage **Low-Rank Adaptation (LoRA)** using Hugging Face's PEFT library to make fine-tuning efficient and feasible on limited GPU resources.

---

## 🧠 Project Overview

This project trains a transformer-based language model to generate radiology **Findings** and **Impression** sections given structured inputs like pathology presence scores (e.g., "Cardiomegaly: 1; Pneumonia: uncertain").

We use:
- **BioGPT** (~1.5B parameters) for its biomedical knowledge.
- **LoRA (Low-Rank Adaptation)** to reduce training cost.
- **Hugging Face Transformers & Datasets** for model and data handling.
- **bitsandbytes** for 8-bit quantization.

The dataset (`mimic_cxr_finetune.jsonl`) consists of ~60,000 prompt-completion pairs.

---

## 🔧 Key Steps

### 1. Dataset Preparation
- Created using a Python script that parses the MIMIC-CXR **CheXpert** CSV and corresponding **report text files**.
- Extracts `Findings` and `Impression` sections using regex.
- Pairs them with diagnostic labels to form prompt-completion pairs.
- Filters for studies with **positive pathology labels**.
- Saves final output to `mimic_cxr_finetune.jsonl`.

#### Example JSONL Record:
```json
{
  "prompt": "Atelectasis: 1; Lung Opacity: 1; Pneumonia: uncertain. \nFindings:",
  "completion": " Since the prior radiograph, there has been no significant change... \nImpression: No significant change since the prior radiograph..."
}
```

---

### 2. Tokenization & Preprocessing
- Tokenizes the prompt + completion (concatenated with a newline) using BioGPT tokenizer.
- Masks out the prompt portion in `labels` (using -100) so **loss is only computed on the report generation**.
- Truncates/pads sequences to a max length of 512.

---

### 3. Model Setup (LoRA)
- Loads BioGPT in **8-bit mode** using `bitsandbytes`.
- Prepares it for training with `prepare_model_for_kbit_training()`.
- Applies **LoRA adapters** using PEFT:
  - `r=8`, `lora_alpha=16`, `dropout=0.1`
  - Applied to `q_proj`, `v_proj` layers.
- Only LoRA weights are trained (~0.1% of total model params).

---

### 4. Training Configuration
- Uses Hugging Face `Trainer` for managing training.
- Key settings:
  - `batch_size=4` with `gradient_accumulation_steps=4`
  - `num_train_epochs=2`
  - `learning_rate=2e-4`, `fp16=True`
- Training and evaluation on the tokenized dataset.

---

### 5. Model Saving
- After training, saves LoRA-adapted model and tokenizer using `save_pretrained()`.
- The saved model includes only adapter weights and references to the base BioGPT.

---

### 6. Inference Example
- Reloads the base model and applies saved LoRA adapters.
- Tokenizes a sample prompt and generates the report using `model.generate()`.
- Decodes the output to display the generated Findings/Impression.

---

## 📁 Files
- `finetune.ipynb` - Full Google Colab-compatible training script.
- `llm_dataset_prep.py` - Script to generate the `mimic_cxr_finetune.jsonl` dataset.
- `mimic_cxr_finetune.jsonl` - Final prompt-completion training dataset (~60k samples).

---

## 📜 References
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/transformers/main/en/peft)
- [BioGPT Model Card](https://huggingface.co/microsoft/biogpt)
- [MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr/2.0.0/)

---


