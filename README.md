# 🧠 Fine-Tuning DeepSeek-R1 for Medical Reasoning (LoRA + Unsloth) <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="120"/>

[![Hugging Face Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue)](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-DeepSeekR1_8B-purple)](https://huggingface.co/deepseek-ai/DeepSeek-R1)


This project fine-tunes the `DeepSeek-R1-Distill-Llama-8B` model on the `medical-o1` dataset using efficient methods like **QLoRA**, **LoRA adapters**, and the **Unsloth framework**. The goal is to enhance clinical reasoning capabilities in medical QA systems while reducing memory and compute requirements.

---

## 🔗 Kaggle Notebook Version
You can view and run this project directly on [Kaggle](https://www.kaggle.com/code/sohamkar529/deepseek2c7cb42855)  

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

## 🧭 How It Works

<p align="center">
  <img src="https://github.com/soham-kar/deepseek/blob/main/lora.png" width="1000 "/>
</p>

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

## 🧬 Dataset Used: [`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)

> 🔍 This dataset is a **supervised fine-tuning (SFT)** dataset specifically designed to evaluate and train LLMs on complex **medical reasoning**, including diagnosis, explanation, and treatment recommendation.

### 📦 Dataset Details

| Field        | Description |
|--------------|-------------|
| 📚 Name       | `medical-o1-reasoning-SFT` |
| 🧪 Source     | [`FreedomIntelligence`](https://huggingface.co/FreedomIntelligence) on Hugging Face |
| 🧠 Focus      | Medical question-answering and reasoning |
| 🧾 Format     | JSONL / Hugging Face Datasets format |
| 🔧 Fields     | `instruction`, `input`, `output` |
| 🔁 Type       | Instruction-tuned (SFT) |
| 📊 Size       | ~10,000 examples (approx.) |
| 🩺 Domain     | Clinical QA, Diagnosis, Medical Education |
| 🗂️ License    | Apache 2.0 |

### 🧾 Sample Entry

```json
{
  "instruction": "Explain the pathophysiology of Type 1 Diabetes.",
  "input": "",
  "output": "Type 1 Diabetes is caused by autoimmune destruction of pancreatic beta cells..."
}
```

### 🤖 Why It Was Chosen

- Emphasizes **chain-of-thought medical reasoning**
- Structured for **instruction tuning**, compatible with LoRA + QLoRA
- Pairs well with models like `DeepSeek-R1` due to its distilled instruction format

### 🧩 Integration in Pipeline

We load and preprocess it using:

```python
from datasets import load_dataset

dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", split="train")
```

And format it to this template:

```
### Instruction:
{instruction}
### Input:
{input}
### Response:
{output}
```

### 🧠 Citation

```bibtex
@misc{freedomintelligence2024medicalo1,
  title={Medical O1 Reasoning Dataset},
  author={FreedomIntelligence},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT}},
}
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

## 🧪 Training Workflow: From Base Model to Domain Expert

This section illustrates the **end-to-end fine-tuning pipeline** using `DeepSeek-R1-Distill-Llama-8B`, the `medical-o1` dataset, and efficient training strategies like LoRA and quantization — all implemented through the `Unsloth` framework.

### 🖼️ Overview Diagram  
![Pipeline](https://github.com/soham-kar/deepseek/blob/main/Medical%20Model%20Fine-Tuning%20Flowchart.png)

---

### 🔁 Step-by-Step Breakdown

| Step | Description |
|------|-------------|
| 🧠 **1. Load Base Model** | Initialize `DeepSeek-R1-Distill-Llama-8B` with **4-bit quantization** using:<br>`load_in_4bit=True`<br>This reduces memory usage drastically while maintaining performance. |
| 🗂️ **2. Load & Format Dataset** | Use the `medical-o1` dataset containing:<br>- `instruction` (task)<br>- `input` (context)<br>- `output` (expected answer)<br><br>All samples are converted to a standard prompt format:  <br>`### Instruction:` … `### Input:` … `### Response:` |
| 🧩 **3. Inject LoRA Adapters** | With `FastLanguageModel.get_peft_model()`, LoRA adapters are inserted into transformer layers.<br>LoRA hyperparameters:<br>`r=16`, `alpha=16`, `dropout=0.05` |
| 🧪 **4. Fine-Tune the Model** | The model is fine-tuned using:<br>✅ **AdamW optimizer**<br>✅ **Linear learning rate scheduler**<br>✅ **3 epochs**, batch size = 2<br>✅ Trained on CUDA (GPU)<br><br>All while only updating LoRA parameters. |
| 💾 **5. Save Adapters** | After training, only the LoRA adapters are saved in `.safetensors` format:<br>`models/adapter_model.safetensors`<br>This is lightweight (~100MB) and reusable. |
| 🧠 **6. Inference & Evaluation** | Perform **pre/post fine-tuning inference** on medical queries:<br>✅ See how model reasoning improves<br>✅ Compare medical accuracy, depth, and relevance |

---

### ⚡ Visual Recap (Mini Flowchart)

```text
┌──────────────┐      ┌────────────────┐      ┌───────────────┐
│  Base Model  │ ──▶ │ Quantize (4-bit)│ ──▶ │ Load Dataset  │
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
```


---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/soham-kar/deepseek.git
cd deepseek
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
## 🧠 Before vs After: Inference Comparison

This section highlights the impact of fine-tuning. The base model produces generic, high-level responses, while the fine-tuned model demonstrates deeper clinical understanding.

| Prompt                        | 🧪 Base Model Response                          | ✅ Fine-Tuned Model Response                                             |
|------------------------------|--------------------------------------------------|--------------------------------------------------------------------------|
| Explain Type 2 Diabetes      | "It is a disease affecting blood sugar..."       | "Type 2 Diabetes is characterized by insulin resistance, where the body's cells do not respond to insulin effectively..." |
| What is insulin resistance?  | "Insulin helps manage blood sugar..."           | "Insulin resistance occurs when muscle, fat, and liver cells fail to respond properly to insulin, leading to hyperglycemia..." |

> ✅ **Observation**: After fine-tuning, the model shows improved domain alignment with accurate medical terminology, structured reasoning, and fewer generic statements.
---

## 💡 Next Steps

- Merge LoRA weights into base model for export
- Build interactive demo with Streamlit/Gradio
- Experiment with `medical-mcqa` and `pubmedqa`

---

