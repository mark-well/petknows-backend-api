import asyncio
import io
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from services.inference import get_embedding_async, load_model_eagerly
from supabase_client import match_pets


load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("pet-identification")


# *"".split(",") returns [''] rather than [], which would silently add a*
# *bogus empty-string CORS origin when ALLOWED_ORIGINS is unset. Filter it out.*
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
allowed_origins = [
    origin.strip()
    for origin in _raw_origins.split(",")
    if origin.strip()
]

# *Distance threshold on L2-normalized embeddings (Euclidean pairwise distance),*
# *matching the best_threshold logged at the end of training in train.py.*
# *NOTE: this is a distance threshold, not a cosine-similarity threshold -- the*
# *two are related but not interchangeable. Update this value whenever the*
# *model is retrained.*
DISTANCE_THRESHOLD = float(
    os.environ.get("MATCH_DISTANCE_THRESHOLD", "0.6")
)

# *Must match the model_version tag written into pet_images rows when their*
# *embeddings were computed. Comparing embeddings from different model*
# *versions is meaningless -- the embedding space itself shifts between*
# *checkpoints. Bump this whenever the deployed model is retrained, and*
# *re-embed pet_images accordingly.*
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v6")

MAX_MATCHES_RETURNED = 3
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # *10 MB*


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
    # *Load the model once at startup instead of lazily on first request, so*
    # *the first caller doesn't pay multi-second model-load latency.*
    logger.info("Starting pet identification API")
    logger.info(
        "Loading model | model_version=%s | distance_threshold=%.3f",
        MODEL_VERSION,
        DISTANCE_THRESHOLD,
    )

    load_model_eagerly()

    logger.info("Model loaded successfully")


async def read_uploaded_image(file: UploadFile) -> Image.Image:
    logger.debug(
        "Reading uploaded image | filename=%s | content_type=%s",
        file.filename,
        file.content_type,
    )

    if file.content_type is None or not file.content_type.startswith("image/"):
        logger.warning(
            "Rejected upload: invalid content type | filename=%s | content_type=%s",
            file.filename,
            file.content_type,
        )
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    contents = await file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        logger.warning(
            "Rejected upload: image too large | filename=%s | size_bytes=%d",
            file.filename,
            len(contents),
        )
        raise HTTPException(
            status_code=400,
            detail="Image too large.",
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()  # *force full decode now, so corrupt files fail here, not later*
    except Exception:
        logger.warning(
            "Rejected upload: could not decode image | filename=%s",
            file.filename,
        )
        raise HTTPException(
            status_code=400,
            detail="Could not read image file.",
        )

    logger.debug(
        "Image loaded successfully | filename=%s | size=%s | format=%s",
        file.filename,
        image.size,
        image.format,
    )

    return image


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    model_version: str


def distance_to_confidence(distance: float, threshold: float) -> float:
    """Maps a raw distance onto an intuitive [0, 1] score: 1.0 at distance=0
    (perfect match), 0.5 exactly at the match threshold, decaying toward 0
    as distance grows past it. NOT a calibrated probability -- would need
    something like Platt scaling against labeled validation pairs for that
    -- but it gives callers a bounded, higher-is-better number instead of
    a raw geometric distance.
    """
    confidence = 1.0 - (distance / (2 * threshold))
    return max(0.0, min(1.0, confidence))


class PetMatch(BaseModel):
    id: str | int | None = None
    name: str | None = None
    distance: float
    confidence: float


class IdentifyResponse(BaseModel):
    found: bool
    message: str
    top_matches: list[PetMatch] = []


@app.get("/")
async def root():
    logger.debug("Health check requested")
    return {"message": "Server is running!"}


@app.post("/get_embedding", response_model=EmbeddingResponse)
async def get_embedding_endpoint(file: UploadFile):
    logger.info(
        "Embedding request | filename=%s",
        file.filename,
    )

    image = await read_uploaded_image(file)
    embedding = await get_embedding_async(image)

    logger.info(
        "Embedding generated | filename=%s | model_version=%s",
        file.filename,
        MODEL_VERSION,
    )

    return EmbeddingResponse(
        embedding=embedding.squeeze().cpu().tolist(),
        model_version=MODEL_VERSION,
    )


@app.post("/identify", response_model=IdentifyResponse)
async def identify_pet(file: UploadFile):
    logger.info(
        "Identification started | filename=%s | model_version=%s | threshold=%.3f",
        file.filename,
        MODEL_VERSION,
        DISTANCE_THRESHOLD,
    )

    try:
        image = await read_uploaded_image(file)

        logger.debug(
            "Generating embedding for identification | filename=%s",
            file.filename,
        )

        current_pet_embedding = await get_embedding_async(image)
        embedding_list = current_pet_embedding.squeeze().cpu().tolist()

        logger.debug(
            "Embedding generated, querying pet matches | filename=%s",
            file.filename,
        )

        # *Matching now happens inside Postgres via the match_pets() SQL function*
        # *(pgvector): for each pet, takes the minimum distance across all of*
        # *that pet's stored images (nearest-image match, not an averaged*
        # *embedding), filtered to the currently deployed model_version, using*
        # *an indexed nearest-neighbor search instead of pulling every embedding*
        # *into Python and looping.*
        matches = await asyncio.to_thread(
            match_pets,
            embedding_list,
            DISTANCE_THRESHOLD,
            MAX_MATCHES_RETURNED,
            MODEL_VERSION,
        )

        # -------------------------------------------------------------------
        # Identification result logging
        # -------------------------------------------------------------------

        if not matches:
            logger.info(
                "Pet NOT FOUND | filename=%s | model_version=%s | threshold=%.3f",
                file.filename,
                MODEL_VERSION,
                DISTANCE_THRESHOLD,
            )

            return IdentifyResponse(
                found=False,
                message="Pet not found.",
            )

        logger.info(
            "Pet FOUND | filename=%s | match_count=%d",
            file.filename,
            len(matches),
        )

        for rank, match in enumerate(matches, start=1):
            distance = match["distance"]
            confidence = distance_to_confidence(
                distance,
                DISTANCE_THRESHOLD,
            )

            logger.info(
                "Match #%d | pet_id=%s | name=%s | distance=%.4f | confidence=%.4f",
                rank,
                match["pet_id"],
                match["name"],
                distance,
                confidence,
            )

        return IdentifyResponse(
            found=True,
            message="Pet found.",
            top_matches=[
                PetMatch(
                    id=match["pet_id"],
                    name=match["name"],
                    distance=match["distance"],
                    confidence=distance_to_confidence(
                        match["distance"],
                        DISTANCE_THRESHOLD,
                    ),
                )
                for match in matches
            ],
        )

    except HTTPException:
        # HTTP errors have already been handled/logged where appropriate.
        raise

    except Exception:
        logger.exception(
            "Identification failed unexpectedly | filename=%s",
            file.filename,
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred while identifying the pet.",
        )