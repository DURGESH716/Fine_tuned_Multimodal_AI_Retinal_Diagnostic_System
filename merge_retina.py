# import torch
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from peft import PeftModel

# BASE_MODEL = "/home/durgesh/VLLM_EngineCore/qwen_model_weights"
# LORA_PATH = "/home/durgesh/VLLM_EngineCore/outputs/final_retina_lora"
# SAVE_PATH = "/home/durgesh/VLLM_EngineCore/qwen_retina_final_merged"

# print("Merging weights... this will create a single 15GB+ model folder.")
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
# )
# model = PeftModel.from_pretrained(model, LORA_PATH)
# merged_model = model.merge_and_unload()

# # Save everything into one folder
# merged_model.save_pretrained(SAVE_PATH)
# processor = AutoProcessor.from_pretrained(BASE_MODEL)
# processor.save_pretrained(SAVE_PATH)
# print(f"Done! Use {SAVE_PATH} for all future work.")


import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# --- CONFIGURATION ---
BASE_MODEL = "/home/durgesh/VLLM_EngineCore/qwen_model_weights"
LORA_PATH = "/home/durgesh/VLLM_EngineCore/outputs/retina_v2_professional/final_retina_lora"
SAVE_PATH = "/home/durgesh/VLLM_EngineCore/qwen_retina_final_merged"

def merge_lora():
    print(f"🚀 Loading base model: {BASE_MODEL}")
    # Load in bfloat16 to match training precision and save RAM
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
    # merge_and_unload turns the LoRA layers into standard linear layers
    merged_model = model.merge_and_unload()

    print(f"Stage 3: Saving merged model and processor to {SAVE_PATH}")
    # Ensure the directory exists
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Save the weights
    merged_model.save_pretrained(SAVE_PATH, safe_serialization=True)
    
    # Save the processor (crucial for vision grid logic)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.save_pretrained(SAVE_PATH)

    print("\n✅ Success!")
    print(f"Merged model is ready. You can now delete the 'outputs' folder to save space.")
    print(f"Final Model Directory: {SAVE_PATH}")

if __name__ == "__main__":
    merge_lora()
