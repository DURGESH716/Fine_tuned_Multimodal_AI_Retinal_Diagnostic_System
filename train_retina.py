import torch
import os
import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# --- CONFIGURATION ---
MODEL_PATH = "/home/durgesh/VLLM_EngineCore/qwen_model_weights"
DATASET_PATH = "/home/durgesh/VLLM_EngineCore/cleaned_retina_train.jsonl"
OUTPUT_DIR = "/home/durgesh/VLLM_EngineCore/outputs/retina_v2_professional"

# --- 1. DATASET CLASS ---
class RetinaDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.processor = processor
        with open(jsonl_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image"]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new('RGB', (448, 448), (0, 0, 0))
        
        # Use a simplified message structure. 
        # The processor will calculate the correct number of vision tokens.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": "Analyze this retinal fundus photograph and provide the diabetic retinopathy grade."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["conversations"][1]["value"]}
                ]
            }
        ]
        
        # apply_chat_template inserts <|vision_start|>, <|image_pad|>, etc.
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # The processor MUST receive the PIL image here to sync with the text pads
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs

def main():
    # 2. MODEL LOADING
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    print(f"Loading Model from {MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )

    # 3. LORA & GRADIENT FIXES
    print("Applying LoRA adapters (Rank 64)...")
    # This line is critical to fix the 'None of the inputs have requires_grad' warning
    model.enable_input_require_grads() 
    
    lora_config = LoraConfig(
        r=64, 
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 4. DATA COLLATOR
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True
    )

    # 5. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,      
        gradient_accumulation_steps=8,     
        learning_rate=1e-4,
        bf16=True,                         
        logging_steps=1,
        max_steps=300,                     
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        lr_scheduler_type="cosine",
        # Set to False initially to troubleshoot stability; use True if VRAM peaks
        gradient_checkpointing=False,
        report_to="none"
    )

    # 6. INITIALIZE TRAINER
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RetinaDataset(DATASET_PATH, processor),
        data_collator=data_collator,
    )

    # 7. START TRAINING
    print(f"🚀 Launching Stable Training on Blackwell...")
    trainer.train()
    
    # 8. SAVE
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_retina_lora"))
    print("Training Complete. LoRA adapters saved.")

if __name__ == "__main__":
    main()