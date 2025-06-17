from fastapi import FastAPI, HTTPException, Header
import logging
import os
import asyncio
import httpx

from pinecone import Pinecone

# === CONFIG ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_rFk4H_GH6HWkbePFmQ5LyuQLMqxN5BJJiaSyhpqSTPjYwt4VKvb3xzXNtkcqWgQCYP6Hc")
INDEX_NAME = "lg-index-gemini"
API_KEY = os.getenv("ORCHESTRATOR_SECRET", "")  # Optional
LAZYBUGURU_ENDPOINT = os.getenv("LAZYBUGURU_ENDPOINT", "http://localhost:7002/scrape-index")
KNOWLEDGEBASE_ENDPOINT = os.getenv("KNOWLEDGEBASE_ENDPOINT", "http://localhost:7003/index-lg-sites")

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "severity": "%(levelname)s", "message": "%(message)s"}'
)

# === INIT ===
app = FastAPI()
pc = Pinecone(api_key=PINECONE_API_KEY)


# === HELPERS ===
def clear_index(index_name: str):
    try:
        if index_name not in pc.list_indexes().names():
            logging.warning(f"Index '{index_name}' not found, skipping deletion.")
            return {"skipped": True, "message": "Index not found."}
        index = pc.Index(index_name)
        index.delete(delete_all=True)
        logging.info(f"✅ Index '{index_name}' cleared.")
        return {"status": "success", "message": f"Index '{index_name}' cleared."}
    except Exception as e:
        logging.error(f"Failed to clear index: {str(e)}")
        raise


async def call_endpoint(name: str, url: str, method: str = "post") -> dict:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if method.lower() == "get":
                response = await client.get(url)
            else:
                response = await client.post(url)
            response.raise_for_status()
            logging.info(f"✅ {name} succeeded: {response.status_code}")
            return {
                "status": "success",
                "data": response.json(),
                "code": response.status_code
            }
    except httpx.HTTPStatusError as e:
        logging.error(f"❌ {name} failed with status {e.response.status_code}: {e.response.text}")
        return {
            "status": "error",
            "code": e.response.status_code,
            "error": e.response.text
        }
    except Exception as e:
        logging.error(f"❌ {name} call failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


# === ORCHESTRATOR ENDPOINT ===
@app.post("/rebuild-index")
async def rebuild_index(x_api_key: str = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    logging.info("🔁 Starting full index rebuild...")

    # Step 1: Clear index
    try:
        clear_result = clear_index(INDEX_NAME)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index clear failed: {str(e)}")

    # Step 2: Wait 10 seconds
    logging.info("⏳ Waiting 5 seconds before continuing...")
    await asyncio.sleep(5)

    # Step 3: Index Lazybuguru (GET)
    logging.info("🚀 Calling Lazybuguru scraper...")
    lazy_result = await call_endpoint("Lazybuguru", LAZYBUGURU_ENDPOINT, method="get")


    # Step 4: Index Knowledgebase (POST)
    logging.info("📚 Calling Knowledgebase indexer...")
    kb_result = await call_endpoint("Knowledgebase", KNOWLEDGEBASE_ENDPOINT, method="post")

    logging.info("✅ Index rebuild complete.")

    return {
        "status": "completed",
        "clear_result": clear_result,
        "lazybuguru_result": lazy_result,
        "knowledgebase_result": kb_result
    }
