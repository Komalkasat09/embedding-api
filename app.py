from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

class InputText(BaseModel):
    text: str
    is_query: bool = False

@app.post("/embed")
def embed(input: InputText):
    # Add prefix if it's a query (per BGE's recommendation)
    text = input.text
    if input.is_query:
        text = "Represent this sentence for searching relevant passages: " + text
    
    embedding = model.encode(text, normalize_embeddings=True).tolist()
    return {"embedding": embedding}
