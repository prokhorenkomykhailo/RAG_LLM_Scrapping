from fastapi import FastAPI, HTTPException, Query
from pinecone import Pinecone
import logging

# === CONFIG ===
PINECONE_API_KEY = 'pcsk_rFk4H_GH6HWkbePFmQ5LyuQLMqxN5BJJiaSyhpqSTPjYwt4VKvb3xzXNtkcqWgQCYP6Hc'
INDEX_NAME = 'lg-index-gemini'

# === INIT ===
pc = Pinecone(api_key=PINECONE_API_KEY)
app = FastAPI()
logging.basicConfig(level=logging.INFO)


# === ROUTE ===
@app.get("/clear-index")
def clear_index(index_name: str = Query(default=INDEX_NAME)):
    try:
        if index_name not in pc.list_indexes().names():
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' does not exist.")

        index = pc.Index(index_name)
        index.delete(delete_all=True)
        logging.info(f"✅ All vectors deleted from '{index_name}'")
        return {"status": "success", "message": f"Index '{index_name}' cleared successfully."}
    except Exception as e:
        logging.error(f"Error clearing index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# === LOCAL RUN ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7001)
