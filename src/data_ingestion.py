import os
import json
import arxiv
from pyspark.sql import SparkSession
from src.config import BASE_DIR, RAW_PDF_DIR

def setup_kaggle_auth(kaggle_json_path="kaggle.json"):
    """Moves kaggle.json to the correct location for authentication."""
    target_dir = os.path.expanduser("~/.kaggle")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # In a real script, ensure kaggle.json exists before moving
    if os.path.exists(kaggle_json_path):
        os.system(f"cp {kaggle_json_path} {target_dir}/")
        os.system(f"chmod 600 {target_dir}/kaggle.json")
        print("Kaggle auth setup complete.")
    else:
        print("Warning: kaggle.json not found in current directory.")

def download_metadata():
    """Downloads ArXiv metadata using Kaggle API."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Downloading ArXiv metadata...")
    api.dataset_download_files('Cornell-University/arxiv', path=BASE_DIR, unzip=True)
    print("Download complete.")

def download_pdfs_from_ids(arxiv_ids):
    """Downloads PDFs using the ArXiv library."""
    print(f"Downloading {len(arxiv_ids)} PDFs...")
    client = arxiv.Client()
    
    # Process in batches to avoid timeouts
    for paper_id in arxiv_ids:
        try:
            paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
            paper.download_pdf(dirpath=RAW_PDF_DIR, filename=f"{paper_id}.pdf")
            print(f"Downloaded: {paper_id}")
        except Exception as e:
            print(f"Failed to download {paper_id}: {e}")

if __name__ == "__main__":
    setup_kaggle_auth()
    # download_metadata() # Uncomment to run
