# 📚 Spark-ArXiv: End-to-End LLM Fine-Tuning Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-3.5-orange?logo=apachespark&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **A scalable MLOps pipeline that ingests scientific papers with PySpark, generates synthetic Q&A datasets using a Teacher LLM, and fine-tunes a specialized Student LLM (Mistral-7B) for domain-specific RAG.**

---

## 🚀 Project Overview

**Spark-ArXiv** is a comprehensive Data Engineering & LLM project designed to solve the challenge of querying dense scientific literature. Unlike simple RAG wrappers, this project implements a full lifecycle pipeline:

1.  **Big Data Ingestion:** Uses **Apache Spark** to process and OCR raw PDF research papers from ArXiv.
2.  **Vector Search:** Builds a semantic search engine using **FAISS** and Sentence Transformers.
3.  **Synthetic Data Factory:** Uses a "Teacher" model (Mistral-7B) to autonomously generate high-quality Q&A pairs from the processed text.
4.  **Fine-Tuning:** Trains a domain-specific "Student" model using **QLoRA** on the synthetic dataset.
5.  **Deployment:** Exposes the final model via a **Gradio** chat interface.

---

## 🏗️ Architecture

The pipeline is divided into two major phases:

### Phase 1: Data Engineering & RAG
* **Source:** ArXiv Metadata (Kaggle) & PDF API.
* **Processing:** PySpark for parallel OCR (PyMuPDF) and Chunking (LangChain).
* **Storage:** Parquet (Metadata) + FAISS (Vector Index).

### Phase 2: LLM Fine-Tuning & Application
* **Generation:** Mistral-7B generates `(Question, Answer, Context)` triplets.
* **Training:** QLoRA Fine-tuning (4-bit quantization) on the synthetic dataset.
* **Inference:** Retrieval-Augmented Generation (RAG) merging vector search with the fine-tuned model.

---

## 📂 Repository Structure

```text
Spark-ArXiv-End-to-End-LLM-Fine-Tuning-Pipeline/
│
├── 📂 data/                   # Local data storage (GitIgnored)
│   ├── raw_pdfs/              # Downloaded papers
│   └── processed/             # Parquet files & FAISS indices
│
├── 📂 src/                    # Source Code
│   ├── config.py              # Central configuration & paths
│   ├── data_ingestion.py      # Kaggle API & PDF downloads
│   ├── text_processor.py      # PySpark OCR & Text Chunking
│   ├── vector_store.py        # Embedding generation & FAISS indexing
│   ├── synthetic_data.py      # "Teacher" model Q&A generation
│   ├── trainer.py             # QLoRA Fine-tuning script
│   └── app.py                 # Gradio Chat Interface
│
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

```

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/AkshitSingh7/Spark-ArXiv.git](https://github.com/AkshitSingh7/Spark-ArXiv.git)
cd Spark-ArXiv

```


2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Kaggle Setup:**
* Place your `kaggle.json` API key in the root directory (or `~/.kaggle/`).
* The ingestion script handles authentication automatically.



---

## 💻 Usage Guide

### 1. Run the Data Pipeline

Download ArXiv metadata and PDFs, then process them with Spark.

```bash
python src/data_ingestion.py
# Followed by processing
python src/text_processor.py

```

### 2. Build Vector Index

Convert processed text chunks into vector embeddings.

```bash
python src/vector_store.py

```

### 3. Generate Synthetic Training Data

Use the "Teacher" model to create a Q&A dataset from your specific papers.

```bash
python src/synthetic_data.py

```

*Output: `data/synthetic_arxiv_qa_dataset.json*`

### 4. Fine-Tune the Model

Train the "Student" model (Mistral-7B) on your synthetic data.

```bash
python src/trainer.py

```

*Output: `models/final_adapter/*`

### 5. Launch the App

Start the Chat UI to query your papers.

```bash
python src/app.py

```

---

## 📊 Performance & Tech Stack

| Component | Technology | Description |
| --- | --- | --- |
| **Orchestration** | Apache Spark | Handles massive PDF processing & OCR at scale. |
| **Embeddings** | `all-MiniLM-L6-v2` | Fast, high-quality sentence embeddings. |
| **Vector DB** | FAISS (CPU) | Efficient similarity search for RAG. |
| **LLM (Base)** | Mistral-7B-v0.1 | State-of-the-art 7B parameter model. |
| **Training** | QLoRA + PEFT | Parameter-Efficient Fine-Tuning on consumer GPUs. |
| **UI** | Gradio | Interactive web interface for model testing. |

---

## 🔮 Future Improvements

* [ ] **Hybrid Search:** Combine BM25 keyword search with FAISS vector search.
* [ ] **Multi-Modal:** Add support for extracting and interpreting charts/images from papers.
* [ ] **Evaluation:** Implement RAGAS metrics to score retrieval accuracy.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📜 License

This project is licensed under the MIT License.

```

```
