# deepseek
# 🧠 Fine-Tuning DeepSeek-R1 for Medical Reasoning (QLoRA + Unsloth)

This project fine-tunes the `DeepSeek-R1-Distill-Llama-8B` model on the `medical-o1` dataset using efficient methods like **QLoRA**, **LoRA adapters**, and the **Unsloth framework**. The goal is to enhance clinical reasoning capabilities in medical QA systems while reducing memory and compute requirements.

---

## 🧬 Techniques Used

### 🔄 LoRA (Low-Rank Adaptation)
LoRA is a parameter-efficient fine-tuning method that injects learnable adapters into frozen model layers. It drastically reduces the number of trainable parameters.

- **Rank (`r`)**: 16  
- **Alpha**: 16  
- **Dropout**: 0.05  
- **Benefits**:
  - No need to update the full base model
  - Smaller memory footprint
  - Faster training, easier adapter sharing (`.safetensors`)

---

### ⚖️ 4-Bit Quantization (QLoRA Style)

Quantization is a technique that reduces the precision of a model’s weights from high-precision formats (like 32-bit float) to lower-precision formats (like 4-bit integers), significantly cutting down memory usage.

In this project, we apply **4-bit quantization** using `load_in_4bit=True` to make it feasible to fine-tune and run an 8B parameter model on limited hardware.

#### 🔍 What Is Quantization?

Large Language Models (LLMs) contain billions of parameters. Storing and manipulating these parameters in full precision (FP32) consumes a huge amount of memory.

**Quantization** addresses this by:
- Replacing high-precision weights (e.g., 32-bit floats) with compact 4-bit integers.
- Storing scaling factors and lookup tables to map back to approximate original values.

Example:
```
Original weight: [0.123456, -0.987654, 1.234567]  # FP32
Quantized:       [6, -8, 12]                      # INT4 with scale
```

#### ✅ Why Are We Using It Here?

| Benefit                      | Explanation |
|-----------------------------|-------------|
| 💾 **Reduced VRAM Usage**   | 4-bit weights use ~8× less memory than FP32 |
| ⚡ **Faster Training/Inference** | Smaller matrices = faster operations |
| 💸 **Lower Compute Cost**   | Enables training on free/Pro Colab GPUs |
| 🤝 **Compatible with LoRA** | Works seamlessly with parameter-efficient fine-tuning |

Without quantization, training a model like `DeepSeek-R1-Distill-Llama-8B` (~8B parameters) would require >24GB of GPU VRAM. Using quantization:

```python
load_in_4bit = True
```

… you enable memory-efficient training with:
- 🐑 [Unsloth](https://github.com/unslothai/unsloth)
- 🚀 LoRA adapters
- 🔋 Limited compute budgets

This technique is part of the **QLoRA** approach, allowing high-performing fine-tuning with low resource requirements.

---

### 🐑 Unsloth Framework

[Unsloth](https://github.com/unslothai/unsloth) is an optimized backend for Hugging Face Transformers that enables efficient training of large language models with LoRA and quantization.

- **Faster downloads**
- **Lower memory usage**
- **Accelerated training loop**

```python
from unsloth import FastLanguageModel
```

---

### 🚀 FastLanguageModel (Unsloth API)

`FastLanguageModel` is an enhanced wrapper for Hugging Face models.

Used for:

- Loading quantized models
- Preparing them for LoRA
- Training with memory-efficient ops

```python
FastLanguageModel.from_pretrained(...)
FastLanguageModel.get_peft_model(...)
FastLanguageModel.prepare_model_for_training(...)
```

---

## 🗂️ Repository Structure

```
.
├── notebooks/
│   └── Fine_Tuning_DEEPSEEK_R1.ipynb     # Full pipeline
├── models/
│   └── adapter_model.safetensors         # LoRA weights only
├── assets/
│   └── pipeline_overview.png             # Image showing full flow
├── README.md
```

---

## 🧪 Training Overview

┌──────────────┐      ┌────────────────┐      ┌───────────────┐
│  Base Model  │ ──▶  │ Quantize (4-bit)│ ──▶ │ Load Dataset  │
└──────────────┘      └────────────────┘      └─────┬─────────┘
                                                    ▼
                                          ┌────────────────────┐
                                          │ Format Prompts     │
                                          └────────┬───────────┘
                                                   ▼
                                         ┌─────────────────────┐
                                         │ Apply LoRA Adapters │
                                         └────────┬────────────┘
                                                  ▼
                                        ┌──────────────────────┐
                                        │ Fine-Tune (3 Epochs) │
                                        └────────┬─────────────┘
                                                 ▼
                                       ┌────────────────────────┐
                                       │ Save Adapter Weights   │
                                       └────────┬───────────────┘
                                                ▼
                                    ┌────────────────────────────┐
                                    │ Post-Tune Medical Inference│
                                    └────────────────────────────┘


![Pipeline](./assets/pipeline_overview.png)

- Load model with `load_in_4bit=True`
- Format medical instruction dataset
- Apply LoRA adapters with Unsloth
- Train for 3 epochs with AdamW
- Save LoRA adapters (`.safetensors`)
- Run post-training inference

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/your-username/deepseek-medical-finetune.git
cd deepseek-medical-finetune
pip install -r requirements.txt
```

<details>
<summary>Dependencies (example)</summary>

```txt
transformers>=4.40.0
accelerate
unsloth
datasets
bitsandbytes
peft
```

</details>

---

## 📘 Inference (After Fine-Tuning)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    load_in_4bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "models/adapter_model.safetensors")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

prompt = "### Instruction:\nExplain the mechanism of insulin resistance.\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## ✅ Results Summary

| Metric | Pre-Fine-Tuning | Post-Fine-Tuning |
|--------|------------------|------------------|
| Medical Accuracy | ❌ Generic | ✅ Specific & Domain-aware |
| Clinical Reasoning | ❌ Surface-Level | ✅ Step-by-Step Reasoning |
| Inference Coherence | ⚠️ Mixed | ✅ Consistent |

---

## 🔗 Kaggle Notebook Version
You can view and run this project directly on [Kaggle](https://www.kaggle.com/code/sohamkar529/deepseek2c7cb42855)  


---

## 💡 Next Steps

- Merge LoRA weights into base model for export
- Build interactive demo with Streamlit/Gradio
- Experiment with `medical-mcqa` and `pubmedqa`

---
