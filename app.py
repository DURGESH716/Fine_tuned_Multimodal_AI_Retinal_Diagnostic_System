import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info
import os

MODEL_PATH = "/home/durgesh/VLLM_EngineCore/qwen_retina_final_merged"

print("🚀 Initializing Retinal Diagnostic System on Blackwell GPU...")

# Use trust_remote_code and specific dtype
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
).eval() # Set to eval mode for inference

def predict_retina(img):
    if img is None:
        return "Please upload an image."

    # Step 1: Force resize to standard Qwen-friendly dimensions
    # This prevents the "vectorized gather kernel index out of bounds"
    image = Image.fromarray(img).convert("RGB").resize((448, 448))
    
    # Step 2: Use exact prompt from your JSONL training data
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Analyze this retinal fundus photograph and provide the diabetic retinopathy grade."}
            ]
        }
    ]

    # Step 3: Process inputs carefully
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    # We pass 'image_inputs' directly to avoid manual rescaling issues
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to("cuda")

    # Step 4: Generation with stable Blackwell settings
    with torch.no_grad():
        try:
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=False, # Use greedy decoding for clinical consistency
                use_cache=False  # Disable cache to avoid the "gather kernel" error
            )
            
            # Decode only the response part
            generated_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response
        except Exception as e:
            return f"Inference Error: {str(e)}. Try restarting the app."

# --- UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ AI Retinal Diagnostic System")
    gr.Markdown("Fine-tuned Qwen2.5-VL for Clinical Diabetic Retinopathy Grading.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Fundus Image")
            btn = gr.Button("Generate Diagnosis", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="Model Findings", lines=10)
            
    btn.click(fn=predict_retina, inputs=input_img, outputs=output_text)

if __name__ == "__main__":
    demo.launch(share=True)