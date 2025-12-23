import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, PARQUET_PATH, FAISS_INDEX_PATH

class ArxivVectorStore:
    def __init__(self):
        print("Loading Embedding Model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata = [] # Store IDs/Titles corresponding to vectors

    def embed_and_save(self, df_pandas):
        """Takes a Pandas DF with 'chunk' column, embeds, and saves."""
        sentences = df_pandas['chunk'].tolist()
        
        print("Generating embeddings...")
        embeddings = self.model.encode(sentences, show_progress_bar=True)
        
        # Save to Parquet (Data + Embeddings)
        df_pandas['embedding'] = list(embeddings)
        df_pandas.to_parquet(PARQUET_PATH)
        print(f"Data saved to {PARQUET_PATH}")
        
        # Build FAISS Index
        self._build_faiss(np.array(embeddings))

    def _build_faiss(self, embedding_matrix):
        dimension = embedding_matrix.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embedding_matrix)
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        print("FAISS index saved.")

    def load_index(self):
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        self.df = pd.read_parquet(PARQUET_PATH)
    
    def search(self, query, k=3):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for idx in indices[0]:
            results.append(self.df.iloc[idx]['chunk'])
        return results
