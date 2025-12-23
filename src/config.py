import os

# --- PATH CONFIGURATION ---
# Check if running in Colab or Local
RUNNING_IN_COLAB = "google.colab" in str(get_ipython()) if "get_ipython" in globals() else False

if RUNNING_IN_COLAB:
    BASE_DIR = "/content/drive/MyDrive/ArxivProject_Nov2025"
else:
    BASE_DIR = os.path.abspath("./data")

# Data Paths
RAW_PDF_DIR = os.path.join(BASE_DIR, "raw_pdfs")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
PARQUET_PATH = os.path.join(PROCESSED_DIR, "embeddings.parquet")
FAISS_INDEX_PATH = os.path.join(PROCESSED_DIR, "arxiv.index")
SYNTHETIC_DATA_PATH = os.path.join(BASE_DIR, "synthetic_arxiv_qa_dataset.json")

# Model Paths
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models/final_adapter")

# --- MODEL CONFIGURATION ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_ID = "mistralai/Mistral-7B-v0.1"

# Create directories if they don't exist
os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
