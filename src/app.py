import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.config import LLM_MODEL_ID, MODEL_OUTPUT_DIR
from src.vector_store import ArxivVectorStore

# Load Resources
print("Loading RAG System...")
vector_store = ArxivVectorStore()
vector_store.load_index()

print("Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, load_in_4bit=True, device_map="auto")

# Load Fine-Tuned Adapters
model = PeftModel.from_pretrained(base_model, MODEL_OUTPUT_DIR)
model.eval()

def chat_pipeline(query, history):
    # 1. Retrieve Context
    docs = vector_store.search(query, k=2)
    context = "\n".join(docs)
    
    # 2. Augment Prompt
    prompt = f"""[INST] You are an ArXiv research assistant. Use the context below to answer.
    
    Context:
    {context}
    
    Question: {query} [/INST]"""
    
    # 3. Generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Basic cleanup to remove prompt from output
    clean_response = response.split("[/INST]")[-1]
    return clean_response

if __name__ == "__main__":
    interface = gr.ChatInterface(fn=chat_pipeline, title="ArXiv RAG Bot")
    interface.launch(share=True)
