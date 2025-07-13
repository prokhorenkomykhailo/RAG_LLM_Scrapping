from fastapi import FastAPI, HTTPException
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import hashlib
import requests
import logging
import uvicorn
import os

# === CONFIG ===
INDEX_NAME = "lg-index-gemini"
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIM = 768
BATCH_SIZE = 50
MAX_WORKERS = 8
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 200
DATA_SOURCE_URL = "https://guru-back.refactoring.dev.gggroup.media/companies/LG/LT/TOP"

# === ENVIRONMENT VARS (Optional for GCP secret config) ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_rFk4H_GH6HWkbePFmQ5LyuQLMqxN5BJJiaSyhpqSTPjYwt4VKvb3xzXNtkcqWgQCYP6Hc")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB8s7vaPasg8I2jJ3ZI7oGIsdjmsJ2okgA")

# === LOGGING (Structured for GCP) ===
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "severity": "%(levelname)s", "message": "%(message)s"}'
)

# === CLIENT INIT ===
genai.configure(api_key=GOOGLE_API_KEY)
embedding_client = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# === FASTAPI INIT ===
app = FastAPI()


class EnhancedWebsiteProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_CHUNK_SIZE,
            chunk_overlap=TEXT_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
        )

    def generate_id(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def format_bonus_info(self, bonus_data: Dict) -> str:
        if not bonus_data:
            return "No bonus information available"
        return (
            f"Bonus Terms: {bonus_data.get('terms', 'N/A')}\n"
            f"Advantages: {', '.join(bonus_data.get('advantages', [])) or 'N/A'}"
        )

    def get_tracker_url(self, website: Dict[str, Any]) -> str:
        for category in website.get("categories", []):
            if isinstance(category, dict) and category.get("category") == "TOP":
                return category.get("url", "")
        return ""

    def process_website(self, website: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            tracker_url = self.get_tracker_url(website)
            metadata = {
                "name": website.get("name", "Unknown"),
                "license": website.get("license", []),
                "status": website.get("status", "Unknown"),
                "products": website.get("products", []),
                "categories": [f"{c.get('category', '')} (Rating: {c.get('rating', 'N/A')})"
                               for c in website.get("categories", [])],
                "min_deposit": website.get("min_deposit", "Unknown"),
                "withdrawal_limit": website.get("monthly_withdrawal_limit", "Unknown"),
                "visitors": website.get("monthly_visitors", "Unknown"),
                "owned_by": website.get("owned_by", "Unknown"),
                "headquarters": website.get("headquarters", "Unknown"),
                "founded": website.get("founded", "Unknown"),
                "accepted_countries": website.get("countries", {}).get("accepted", []),
                "restricted_countries": website.get("countries", {}).get("restricted", []),
                "crypto_payments": website.get("payments", {}).get("crypto", []),
                "traditional_payments": website.get("payments", {}).get("traditional", []),
                "providers": website.get("providers", []),
                "bonus_terms": self.format_bonus_info(website.get("bonus", {})),
                "customer_support_languages": website.get("languages", {}).get("customer_support", []),
                "website_languages": website.get("languages", {}).get("website", []),
                "tracker_url": tracker_url
            }

            text_parts = [
                f"## {metadata['name']}",
                f"- Tracker Url: {metadata['tracker_url']}",
                f"**Description**: {website.get('about', 'No description available')}",
                f"**License**: {', '.join(metadata['license'])}",
                f"**Products Offered**: {', '.join(metadata['products'])}",
                f"**Founded**: {metadata['founded']}",
                f"**Min Deposit**: {metadata['min_deposit']}",
                f"**Withdraw Limit**: {metadata['withdrawal_limit']}",
                f"**Visitors**: {metadata['visitors']}",
                f"**Crypto Payments**: {', '.join(metadata['crypto_payments'][:10])}",
                f"**Traditional Payments**: {', '.join(metadata['traditional_payments'][:10])}",
                f"**Providers**: {', '.join(metadata['providers'][:15])}",
                f"**Bonus Info**: {metadata['bonus_terms']}",
                f"**Country Access**: {', '.join(metadata['accepted_countries'][:10])}",
                f"**Country Blocked**: {', '.join(metadata['restricted_countries'][:10])}",
                f"**Support Langs**: {', '.join(metadata['customer_support_languages'])}",
                f"**Website Langs**: {', '.join(metadata['website_languages'])}",
            ]

            full_text = "\n".join(filter(None, text_parts))
            chunks = self.text_splitter.split_text(full_text)

            return [{
                "id": f"{self.generate_id(metadata['name'])}-chunk{i}",
                "text": chunk,
                "metadata": {
                    "name": metadata["name"],
                    "tracker_url": metadata["tracker_url"]
                }
            } for i, chunk in enumerate(chunks)]

        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return []


class OptimizedPineconeIndexer:
    def __init__(self):
        self.index = self._init_index()

    def _init_index(self):
        if INDEX_NAME not in pinecone.list_indexes().names():
            pinecone.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return pinecone.Index(INDEX_NAME)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return embedding_client.embed_documents(texts)

    def index_batch(self, batch: List[Dict[str, Any]]):
        texts = [item["text"] for item in batch]
        embeddings = self.generate_embeddings(texts)
        vectors = [{
            "id": item["id"],
            "values": emb,
            "metadata": {**item["metadata"], "text": item["text"]}
        } for item, emb in zip(batch, embeddings)]

        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[i:i + 100])
        logging.info(f"Indexed batch of {len(batch)} chunks")


def fetch_website_data() -> List[Dict[str, Any]]:
    try:
        resp = requests.get(DATA_SOURCE_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Safely handle variations in structure
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "websites" in data:
            return data["websites"]
        else:
            logging.error(f"Unexpected API format: {type(data)}")
            return []

    except Exception as e:
        logging.error(f"Fetch failed: {str(e)}")
        return []


@app.get("/index-lg-sites")
def index_lg_sites():
    websites = fetch_website_data()
    if not websites:
        raise HTTPException(status_code=500, detail="No websites to index.")

    processor = EnhancedWebsiteProcessor()
    indexer = OptimizedPineconeIndexer()
    all_chunks = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(processor.process_website, site) for site in websites]
        for f in futures:
            try:
                all_chunks.extend(f.result())
            except Exception as e:
                logging.error(f"Future failed: {str(e)}")

    for i in range(0, len(all_chunks), BATCH_SIZE):
        indexer.index_batch(all_chunks[i:i + BATCH_SIZE])

    logging.info(f"✅ Completed indexing {len(all_chunks)} chunks")
    return {"status": "success", "chunks_indexed": len(all_chunks)}


# === LOCAL RUN ===
if __name__ == "__main__":
    uvicorn.run("indexing_kb:app", host="0.0.0.0", port=7003, reload=True)
