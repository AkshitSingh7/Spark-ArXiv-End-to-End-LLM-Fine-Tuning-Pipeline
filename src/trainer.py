import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from src.config import LLM_MODEL_ID, MODEL_OUTPUT_DIR, SYNTHETIC_DATA_PATH

def fine_tune():
    # Load Dataset
    dataset = load_dataset("json", data_files=SYNTHETIC_DATA_PATH, split="train")
    
    # Model Config
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    # LoRA Config
    peft_config = LoraConfig(
        r=64, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    # Trainer
    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=50,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="raw_response", # Column name in JSON
        peft_config=peft_config,
        args=args
    )
    
    trainer.train()
    trainer.model.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Model saved to {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    fine_tune()
