# 👁️ AI Retinal Diagnostic System: End-to-End System

### **Architecture:** Fine-tuned Qwen2.5-VL-7B (Vision-Language Model)

### **Methodology:** AI-Led Knowledge Distillation & QLoRA Fine-Tuning


## 🎯 Problem Statement

Diabetic Retinopathy (DR) remains a leading cause of vision loss globally. Traditional Computer-Aided Diagnosis (CAD) systems typically act as "black boxes," providing only a numerical grade (0-4) without clinical justification.

**Our Solution:** We developed an explainable AI system using a **Vision-Language Model (VLM)**. Unlike standard CNNs, this system identifies specific clinical features (microaneurysms, hemorrhages, exudates) in natural language and provides an automated grade according to the International Clinical Diabetic Retinopathy Severity Scale.


## 📊 Data Information & Distillation

The project utilizes retinal fundus photographs preprocessed via **Gaussian Filtered** normalization to standardize lighting and enhance vascular contrast.

### **Knowledge Distillation for JSONL Generation**

To achieve high-quality clinical reasoning in a smaller model, we employed a **Distillation Workflow**:

* **Teacher Model:** A high-parameter foundation model analyzed raw fundus images to generate detailed medical descriptions.
* **Structured Distillation:** These descriptive outputs were cleaned and converted into a Visual Question Answering (VQA) format.
* **Training Set:** The final `retina_distilled_dataset.jsonl` contains thousands of pairs of *Image + Expert-Level reasoning + Clinical Grade*.


## Model Architecture

<p align="center">
<img width="916" height="600" alt="Model" src="https://github.com/user-attachments/assets/aa6537d2-6bce-4146-aa68-1b20df737eab" />





<em>Figure 1: Unified Multimodal Architecture featuring Vision Encoder, Adapter, and Language Backbone.</em>
</p>

1. **Vision Encoder:** A specialized Vision Transformer (ViT) that handles native image resolutions using 2D-RoPE.
2. **Multimodal Adapter:** A projection layer (MLP) that aligns visual tokens with the language embedding space.
3. **Language Backbone:** A transformer-based LLM optimized for medical reasoning.


## ⚙️ Training Parameters (QLoRA)

We implemented **QLoRA (Quantized Low-Rank Adaptation)** to enable high-fidelity fine-tuning.

| Parameter | Value | Description |
| --- | --- | --- |
| **Rank ()** | 16 | Dimension of the low-rank update matrices. |
| **LoRA Alpha ()** | 32 | Scaling factor for the adapter weights. |
| **Quantization** | 4-bit NF4 | NormalFloat4 quantization for base weights. |
| **Compute Dtype** | `bfloat16` | Maintains precision on high-end Tensor Cores. |
| **Target Modules** | `q_proj, v_proj` | Targeted key attention projections. |

<p align="center">
<img width="692" height="921" alt="Training Dashboard" src="https://github.com/user-attachments/assets/3b5db490-36ab-451b-90ad-127a9c1b603d" />




<em> Screenshot 1: AI Retinal Diagnostic System Dashboard </em>
</p>


## Directory Structure

```text
.
├── assets/                    # Diagrams and UI screenshots
├── gaussian_filtered_images/  # Preprocessed training data (0-4)
├── qwen_retina_final_merged/  # Final weights (Merged LoRA + Base)
├── app.py                        # Gradio UI for diagnostic inference
├── train_retina.py               # Fine-tuning script (PEFT/QLoRA)
├── merge_retina.py               # Script to merge LoRA into Base model
├── retina_distilled_dataset.jsonl  # Distilled VQA-style dataset
└── requirements.txt                # Environment dependencies

```

## 🚀 Getting Started (Replication)

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/retina-ai-diagnostic.git
cd retina-ai-diagnostic

```

### 2. Setup Environment

```bash
pip install -r requirements.txt

```

### 3. Run the Diagnostic App

```bash
python app.py

```
### 📥 Initial Model Setup & First Run

The first time you execute the application, the system will automatically fetch the model weights from the Hugging Face Hub.

#### **1. Automatic Download (First Run Only)**

When you run `python app.py`, the `transformers` library will detect that the model is missing and begin a download (~15GB for the 7B model).

* **Requirement:** Ensure you have at least 20GB of free disk space.
* **Storage Path:** By default, weights are stored in `~/.cache/huggingface/`.

#### **2. Update Model Path (For Local Weights)**

If you have already merged your LoRA weights into a local folder (e.g., `qwen_retina_final_merged`), you **must** update the path in `app.py` to avoid a fresh download:

```python
# Open app.py and locate the model loading section:
model_name = "./qwen_retina_final_merged"  # Change this from the HF ID to your local path
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa", # Required for Blackwell stability
    device_map="auto"
)

```

## Challenges & Solutions

* **Mathematical Stability:** Encountered numerical instabilities during matrix multiplications; resolved by switching to **Scaled Dot Product Attention (SDPA)**.
* **Spatial Alignment:** Addressed "Device-side Assertion" errors by enforcing a strict **448x448** pixel grid to align with pre-trained vision patches.
* **Numerical Overflow:** Standard FP16 training caused gradient issues; transitioned to **BFloat16** for wider dynamic range support.
* **Contextual Reasoning:** Initial outputs were overly concise; implemented **System Prompt Engineering** to force lesion description before grading.
* **Hardware Constraints:** Full fine-tuning was memory-prohibitive; utilized **QLoRA** to reduce VRAM requirements by over 60%.


## 🔮 Future Scope

* **Attention Heatmaps:** Integrating visual cross-attention maps to highlight retinal lesions.
* **Longitudinal Tracking:** Developing features to track disease progression across multiple visits.
* **Multi-Modal Fusion:** Incorporating systemic patient data (HbA1c, age) for a holistic risk profile.
* **Edge Optimization:** Quantizing the model for real-time deployment on portable screening devices.
* **Automated Reporting:** Generating standardized PDF medical reports for physician sign-off.
