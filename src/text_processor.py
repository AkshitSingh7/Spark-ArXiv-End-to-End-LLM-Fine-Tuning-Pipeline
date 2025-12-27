import fitz  # PyMuPDF
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, ArrayType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import RAW_PDF_DIR

def get_spark_session():
    return SparkSession.builder \
        .appName("ArxivProcessor") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

# --- UDFs for Spark ---

def extract_text_from_pdf(pdf_filename):
    """Opens a PDF and extracts text."""
    try:
        path = f"{RAW_PDF_DIR}/{pdf_filename}"
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception:
        return ""

def chunk_text(text):
    """Splits text into chunks using LangChain."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# Register UDFs
extract_text_udf = udf(extract_text_from_pdf, StringType())
chunk_text_udf = udf(chunk_text, ArrayType(StringType()))

def process_data(spark, metadata_path):
 
    df = spark.read.json(metadata_path)
    
    # Filter for CS category (as per your notebook)
    df_cs = df.filter(col("categories").contains("cs.AI")).limit(100) # Adjust limit
    
    # Extract Text
    df_with_text = df_cs.withColumn("full_text", extract_text_udf(col("id")))
    
    # Chunk Text (Explode creates a new row for each chunk)
    df_chunks = df_with_text.withColumn("chunk", chunk_text_udf(col("full_text")))
    df_final = df_chunks.select("id", "title", "chunk") # explode("chunk") logic usually goes here
    
    return df_final
