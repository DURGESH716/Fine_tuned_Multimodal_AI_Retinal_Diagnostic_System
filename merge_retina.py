import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE_MODEL = "/home/durgesh/VLLM_EngineCore/qwen_model_weights"
LORA_PATH = "/home/durgesh/VLLM_EngineCore/outputs/retina_v2_professional/final_retina_lora"
SAVE_PATH = "/home/durgesh/VLLM_EngineCore/qwen_retina_final_merged"

def merge_lora():
    print(f"🚀 Loading base model: {BASE_MODEL}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu",
        trust_remote_code=True
    )

    print(f"Stage 1: Attaching LoRA adapters from {LORA_PATH}")
    model = PeftModel.from_pretrained(
        model, 
        LORA_PATH,
        torch_dtype=torch.bfloat16
    )

    print("Stage 2: Merging weights into base model (this may take a few minutes)...")

    merged_model = model.merge_and_unload()

    print(f"Stage 3: Saving merged model and processor to {SAVE_PATH}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    

    merged_model.save_pretrained(SAVE_PATH, safe_serialization=True)
    

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.save_pretrained(SAVE_PATH)

    print("\n✅ Success!")
    print(f"Merged model is ready. You can now delete the 'outputs' folder to save space.")
    print(f"Final Model Directory: {SAVE_PATH}")

if __name__ == "__main__":
    merge_lora()
