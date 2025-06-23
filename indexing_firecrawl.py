import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import google.generativeai as genai
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from firecrawl import AsyncFirecrawlApp, ScrapeOptions
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# === CONFIG ===
FIRECRAWL_API_KEY = 'fc-e4019139d9ce482bb1861f15dd13dc0b'
PINECONE_API_KEY = 'pcsk_rFk4H_GH6HWkbePFmQ5LyuQLMqxN5BJJiaSyhpqSTPjYwt4VKvb3xzXNtkcqWgQCYP6Hc'
GOOGLE_API_KEY = 'AIzaSyB8s7vaPasg8I2jJ3ZI7oGIsdjmsJ2okgA'
INDEX_NAME = 'lg-index-gemini'
EMBEDDING_MODEL = 'models/text-embedding-004'
EMBEDDING_DIM = 768
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 200
BATCH_SIZE = 50

# === INIT SERVICES ===
genai.configure(api_key=GOOGLE_API_KEY)
embedding_client = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# === FASTAPI APP ===
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# === HELPERS ===
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class RecursiveSplitter:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_CHUNK_SIZE,
            chunk_overlap=TEXT_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
        )


class MarkdownArticleProcessor(RecursiveSplitter):
    def generate_id(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def process_markdown_article(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            markdown_text = article.get("markdown", "")
            metadata = article.get("metadata", {})
            source_url = metadata.get("og:url") or metadata.get("ogUrl") or metadata.get("url") or "Unknown"
            metadata_text = (
                f"Title: {metadata.get('title') or metadata.get('ogTitle') or 'Untitled'}\n"
                f"Description: {metadata.get('description') or metadata.get('ogDescription') or ''}\n"
                f"URL: {source_url}\n"
                f"Language: {metadata.get('language', 'lt-LT')}\n"
                f"Published: {metadata.get('article:published_time', '')}\n"
                f"Modified: {metadata.get('article:modified_time', '')}\n"
                f"Source: lazybuguru.lt\n"
            )
            full_text = metadata_text + "\n\n" + markdown_text
            chunks = self.text_splitter.split_text(full_text)
            return [{
                "id": f"{self.generate_id(metadata.get('title') or 'Untitled')}-chunk{i}",
                "text": chunk,
                "metadata": metadata
            } for i, chunk in enumerate(chunks)]
        except Exception as e:
            logging.error(f"Error processing article: {str(e)}")
            return []


class PineconeIndexer:
    def __init__(self):
        self.index = self.initialize_index()

    def initialize_index(self):
        if INDEX_NAME not in pinecone.list_indexes().names():
            pinecone.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return pinecone.Index(INDEX_NAME)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            return embedding_client.embed_documents(texts)
        except Exception as e:
            logging.error(f"Embedding error: {str(e)}")
            return []

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [item["text"] for item in batch]
            embeddings = self.generate_embeddings(texts)
            if not embeddings:
                continue

            vectors = []
            for item, emb in zip(batch, embeddings):
                metadata = item.get("metadata", {})

                # Keep only essential lightweight metadata
                compact_metadata = {
                    "title": metadata.get("title") or metadata.get("ogTitle"),
                    "url": metadata.get("og:url") or metadata.get("url"),
                    "language": metadata.get("language", "lt-LT"),
                    "published": metadata.get("article:published_time", ""),
                    "source": "lazybuguru.lt"
                }

                # Add a reference field for full text if needed
                compact_metadata["text_ref"] = f"chunk_{item['id']}"  # Optional, replace with actual ID logic if needed

                vectors.append({
                    "id": item["id"],
                    "values": emb,
                    "metadata": compact_metadata  # ⚠️ exclude full 'text'
                })

            self.index.upsert(vectors=vectors)

        logging.info(f"✅ Indexed {len(chunks)} chunks")

# === ROUTE ===
@app.get("/scrape-index")
async def scrape_and_index():
    firecrawl_app = AsyncFirecrawlApp(api_key=FIRECRAWL_API_KEY)
    markdown_processor = MarkdownArticleProcessor()
    indexer = PineconeIndexer()

    fixed_url = "http://lazybuguru.lt/"  

    try:
        response = await firecrawl_app.crawl_url(
            url=fixed_url,
            limit=25,
            scrape_options=ScrapeOptions(formats=["markdown"], onlyMainContent=True)
        )
        articles = response.model_dump().get("data", [])
        all_chunks = []
        for article in articles:
            all_chunks.extend(markdown_processor.process_markdown_article(article))

        indexer.index_chunks(all_chunks)
        return {"status": "success", "chunks_indexed": len(all_chunks)}
    except Exception as e:
        logging.error(f"Failed to scrape/index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# === RUN LOCALLY ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7002)
