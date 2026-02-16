import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info
import os

MODEL_PATH = "/home/durgesh/VLLM_EngineCore/qwen_retina_final_merged"

print("🚀 Initializing Retinal Diagnostic System on Blackwell GPU...")

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
).eval()

def predict_retina(img):
    if img is None:
        return "Please upload an image."


    image = Image.fromarray(img).convert("RGB").resize((448, 448))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Analyze this retinal fundus photograph and provide the diabetic retinopathy grade."}
            ]
        }
    ]


    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text], 
        images=image_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        try:
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=False, 
                use_cache=False  
            )
            

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
