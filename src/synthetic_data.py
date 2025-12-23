import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.config import LLM_MODEL_ID, PARQUET_PATH, SYNTHETIC_DATA_PATH

def generate_synthetic_data():
    # Load processed chunks
    df = pd.read_parquet(PARQUET_PATH)
    chunks = df['chunk'].sample(50).tolist() # Sample 50 chunks for data generation
    
    # Load Teacher Model (Mistral)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    
    qa_pairs = []
    
    prompt_template = """
    You are a professor. Create a technical question and a correct answer based ONLY on the following text snippet.
    Format your response as a JSON object: {"question": "...", "answer": "..."}
    
    Text: {text}
    """
    
    print("Generating Synthetic Q&A...")
    for text in chunks:
        input_text = prompt_template.format(text=text[:1000]) # Limit length
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # In a real script, use json_repair or regex here to parse response
        qa_pairs.append({"raw_response": response, "context": text})

    # Save
    with open(SYNTHETIC_DATA_PATH, "w") as f:
        json.dump(qa_pairs, f)
    print(f"Saved {len(qa_pairs)} pairs to {SYNTHETIC_DATA_PATH}")

if __name__ == "__main__":
    generate_synthetic_data()
