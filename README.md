# 🎓 Spark-ArXiv: End-to-End LLM Fine-Tuning Pipeline
An end-to-end LLM fine-tuning pipeline using PySpark and Mistral-7B to create a domain-specific research assistant.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5-orange)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Spark-ArXiv** is a scalable machine learning pipeline that demonstrates the complete lifecycle of building a domain-specific Large Language Model (LLM). It ingests raw research papers from ArXiv using **Apache Spark**, generates synthetic training data using a Teacher-Student architecture, and fine-tunes a **Mistral-7B** model to become a subject matter expert in Computer Science research.

## 🚀 Project Overview

The goal of this project was to bridge the gap between **Big Data processing** and **Generative AI**. Instead of relying on generic pre-trained models, this pipeline creates a specialized "Student" model capable of answering complex technical questions based on the latest 2024 CS/ML papers.

### Key Features
* **Scalable Data Ingestion:** Uses **PySpark** to process metadata for 2.8M+ ArXiv papers and filter for relevant Machine Learning (cs.LG) research.
* **PDF ETL Pipeline:** Custom Spark UDFs (User Defined Functions) utilizing **PyMuPDF** to extract and clean text from raw PDF binaries at scale.
* **Semantic Search (RAG):** Implements **FAISS** vector indexing with `sentence-transformers` for efficient retrieval of research context.
* **Synthetic Data Factory:** A "Teacher" model (Mistral-7B-Instruct) generates high-quality, JSON-formatted Q&A pairs from raw text chunks.
* **Efficient Fine-Tuning:** Uses **QLoRA** (Quantized Low-Rank Adaptation) to fine-tune a 4-bit quantized model on a single T4 GPU.
* **Interactive UI:** A **Gradio** chat interface to interact with the fine-tuned model in real-time.

## 🛠️ Tech Stack

* **Data Processing:** Apache Spark (PySpark), Pandas
* **LLMs & Training:** Hugging Face Transformers, `trl` (Transformer Reinforcement Learning), `peft`, `bitsandbytes`
* **Vector Database:** FAISS, Sentence-Transformers
* **Infrastructure:** Google Colab (T4 GPU), Google Drive Integration
* **Visualization/UI:** Gradio, Tqdm

## 🏗️ Architecture Pipeline

1.  **Ingestion:** Download filtered ArXiv metadata (JSON) and PDFs using the Kaggle API.
2.  **Processing:** * Load PDF binaries into Spark DataFrames.
    * Extract text -> Clean -> Chunk into 1000-character segments.
3.  **Embedding:** Generate embeddings for 80,000+ chunks and build a FAISS index.
4.  **Generation (Teacher):** * Prompt Mistral-7B to act as a "Research Assistant".
    * Generate `{"question": "...", "answer": "..."}` pairs from text chunks.
    * *Technique used:* One-shot prompting with `json_repair` for robustness.
5.  **Fine-Tuning (Student):** * Train a new LoRA adapter on the synthetic dataset.
    * Optimization: 4-bit quantization (NF4) and Paged AdamW.
6.  **Inference:** Merge adapter weights and serve via Gradio.

## 💻 Installation & Usage

Since this project relies on GPU acceleration and high-memory Spark sessions, it is optimized for **Google Colab**.

### Prerequisites
* Google Account (for Colab & Drive)
* Hugging Face Access Token
* Kaggle API Key (`kaggle.json`)

### Step-by-Step Run Guide

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/spark-arxiv-llm.git](https://github.com/yourusername/spark-arxiv-llm.git)
    ```

2.  **Setup Environment (Colab)**
    Open the notebook `Spark_ArXiv_Pipeline.ipynb` in Google Colab. Ensure the runtime is set to **T4 GPU**.

3.  **Install Dependencies**
    The notebook automatically handles library installation:
    ```python
    !pip install pyspark kaggle arxiv pymupdf transformers peft bitsandbytes trl gradio json_repair
    ```

4.  **Run the Pipeline**
    * **Phase 1:** Execute Spark cells to ingest PDFs and create `embeddings.parquet`.
    * **Phase 2:** Run the Teacher model loop to generate `synthetic_dataset.json`.
    * **Phase 3:** Execute the SFTTrainer to fine-tune the model.

5.  **Chat with your Model**
    The final cell launches a public Gradio link:
    ```python
    demo.launch(share=True)
    ```

## 📊 Results

* **Data Processed:** ~1,000 PDFs converted to 87,000+ text chunks.
* **Training Efficiency:** Fine-tuning 200 high-quality samples took <10 minutes on T4 GPU.
* **Output Quality:** The Student model successfully learned to adopt the domain-specific jargon and answering style of the source text, outperforming the base model in specific context retrieval tasks.

## 🔮 Future Improvements

* **Scale Up:** Run the generation loop on the full 87k chunk dataset (requires multi-GPU setup).
* **Evaluation:** Implement RAGAS (RAG Assessment) scores to quantitatively measure Hallucination and Answer Relevance.
* **Deployment:** Containerize the merged model using Docker for deployment on AWS SageMaker or EC2.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](issues/).

## 📝 License

This project is [MIT](LICENSE) licensed.

---
*Created by [Your Name]*
