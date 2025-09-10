# 🚀 Fine-Tuning LLMs with LoRA (PEFT)

This repo shows how to fine-tune large language models (LLMs) like **Falcon-7B** or **Mistral-7B** using **LoRA adapters** for parameter-efficient training.

---

## 📌 What is LoRA?
- LLMs = billions of parameters → too expensive to train fully.
- **LoRA** = adds small adapter layers, only those are trained.
- Result → **cheap, fast fine-tuning** on consumer GPUs/Colab.

---

## 🛠️ Tech Stack
- [🤗 Transformers](https://huggingface.co/docs/transformers)
- [🤗 PEFT](https://huggingface.co/docs/peft)
- [🤗 Datasets](https://huggingface.co/docs/datasets)
- [Colab / GPU]

---

## 📂 Repo Structure
- `notebooks/train_lora.ipynb` → quick Colab fine-tune demo  
- `src/train.py` → script for training  
- `data/sample.json` → toy dataset for testing  

---

## ▶️ Quick Start (Colab)

1. Install deps:
```bash
pip install -r requirements.txt
