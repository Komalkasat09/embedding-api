import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingRequest(BaseModel):
    # This now correctly accepts a LIST of texts
    texts: List[str] = Field(..., description="A list of texts to be embedded.")
    is_query: bool = Field(False, description="Whether to add the query prefix.")

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

# Make sure this model name is the one you actually want to use.
# e.g., "BAAI/bge-large-en-v1.5" or "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = "/var/data/huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
model: SentenceTransformer = None

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)

@app.post("/embed", response_model=EmbeddingResponse)
def get_embeddings(request: EmbeddingRequest):
    texts_to_embed = request.texts
    
    if request.is_query:
        # BGE models use a prefix for queries, but all-MiniLM does not.
        # If using BGE, uncomment the line below.
        # texts_to_embed = ["Represent this sentence for searching relevant passages: " + text for text in texts_to_embed]
        pass
        
    embeddings = model.encode(texts_to_embed, normalize_embeddings=True).tolist()
    
    return EmbeddingResponse(embeddings=embeddings)