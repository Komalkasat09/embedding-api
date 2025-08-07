import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from sentence_transformers import SentenceTransformer

# --- Pydantic Models ---
class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="A list of texts to be embedded.")
    is_query: bool = Field(False, description="Whether to add the query prefix.")

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

# --- Model Loading ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# REMOVED: No longer defining a CACHE_DIR because we don't have a persistent disk.
# REMOVED: No longer calling os.makedirs.

model: SentenceTransformer = None

app = FastAPI()

@app.on_event("startup")
def load_model():
    """
    Load the Sentence Transformer model during application startup.
    It will be downloaded to a temporary directory.
    """
    global model
    # REMOVED: The 'cache_folder' argument is gone. The library handles it.
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model '{MODEL_NAME}' loaded successfully into temporary memory.")

@app.post("/embed", response_model=EmbeddingResponse)
def get_embeddings(request: EmbeddingRequest):
    embeddings = model.encode(request.texts, normalize_embeddings=True).tolist()
    return EmbeddingResponse(embeddings=embeddings)