import ast
import asyncio
import io
import os

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from services.inference import get_embedding_async, load_model_eagerly
from supabase_client import get_all_pets

load_dotenv()

# "".split(",") returns [''] rather than [], which would silently add a
# bogus empty-string CORS origin when ALLOWED_ORIGINS is unset. Filter it out.
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
allowed_origins = [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]

# Distance threshold on L2-normalized embeddings (Euclidean pairwise distance),
# matching the best_threshold logged at the end of training in train.py.
# NOTE: this is a distance threshold, not a cosine-similarity threshold -- the
# two are related but not interchangeable. Update this value whenever the
# model is retrained.
DISTANCE_THRESHOLD = float(os.environ.get("MATCH_DISTANCE_THRESHOLD", "0.6"))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # Load the model once at startup instead of lazily on first request, so
    # the first caller doesn't pay multi-second model-load latency.
    load_model_eagerly()


async def read_uploaded_image(file: UploadFile) -> Image.Image:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image too large.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()  # force full decode now, so corrupt files fail here, not later
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    return image


class EmbeddingResponse(BaseModel):
    embedding: list[float]


class PetMatch(BaseModel):
    id: str | int | None = None
    name: str | None = None
    distance: float
    confidence: float


class IdentifyResponse(BaseModel):
    found: bool
    message: str
    top_matches: list[PetMatch] = []

def distance_to_confidence(distance: float, threshold: float) -> float:
    """1.0 at distance=0 (perfect match), 0.5 at the decision threshold,
    approaching 0 as distance grows past the threshold."""
    confidence = 1.0 - (distance / (2 * threshold))
    return max(0.0, min(1.0, confidence))

@app.get("/")
async def root():
    return {"message": "Server is running!"}


@app.post("/get_embedding", response_model=EmbeddingResponse)
async def get_embedding_endpoint(file: UploadFile):
    image = await read_uploaded_image(file)
    embedding = await get_embedding_async(image)
    return EmbeddingResponse(embedding=embedding.squeeze().cpu().tolist())


@app.post("/identify", response_model=IdentifyResponse)
async def identify_pet(file: UploadFile):
    image = await read_uploaded_image(file)
    current_pet_embedding = await get_embedding_async(image)

    # Offload the blocking Supabase call so it doesn't stall the event loop
    # (and every other in-flight request) while waiting on the network/DB.
    pets = (await asyncio.to_thread(get_all_pets)).data

    results = []
    for pet in pets:
        try:
            pet_embedding = torch.from_numpy(
                np.array(ast.literal_eval(pet["embedding"]), dtype=np.float32)
            )
        except (ValueError, SyntaxError, KeyError, TypeError):
            # Skip malformed rows instead of failing the whole request.
            continue

        distance = torch.nn.functional.pairwise_distance(
            current_pet_embedding, pet_embedding.unsqueeze(0)
        ).item()

        results.append({"pet": pet, "distance": distance, "confidence": distance_to_confidence(distance, DISTANCE_THRESHOLD)})

    # Smaller distance = more similar (opposite ordering from cosine similarity).
    results.sort(key=lambda r: r["distance"])
    results = [r for r in results if r["distance"] < DISTANCE_THRESHOLD]
    top_matches = results[:3]

    if not top_matches:
        return IdentifyResponse(found=False, message="Pet not found.")

    return IdentifyResponse(
        found=True,
        message="Pet found.",
        top_matches=[
            PetMatch(
                id=r["pet"].get("id"),
                name=r["pet"].get("name"),
                distance=r["distance"],
                confidence=r["confidence"]
            )
            for r in top_matches
        ],
    )