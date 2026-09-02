import asyncio
import threading
import os
from dotenv import load_dotenv

import torch
from PIL import Image

from datasets.data_transformer import test_transform
from siamese import SiamseNetwork

load_dotenv()

MODEL_VERSION = os.environ.get("MODEL_VERSION")
MODEL_PATH = f"models/{os.environ.get('MODEL')}.pth";

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_model_lock = threading.Lock()


def get_model():
    """Thread-safe lazy singleton. Prefer calling load_model_eagerly() once
    at app startup so the first inference request doesn't pay load latency,
    and so two concurrent early requests can't both trigger a load."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:  # re-check inside the lock (double-checked locking)
                loaded_model = SiamseNetwork().to(device)
                loaded_model.load_state_dict(
                    torch.load(MODEL_PATH, map_location=device)
                )
                loaded_model.eval()
                _model = loaded_model

    return _model


def load_model_eagerly():
    """Call once at app startup to force the model to load immediately."""
    get_model()


def load_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    tensor = test_transform(image)
    return tensor.unsqueeze(0)


def get_embedding(image: Image.Image) -> torch.Tensor:
    model = get_model()
    img = load_image(image)
    with torch.no_grad():
        return model.get_embedding(img.to(device))


async def get_embedding_async(image: Image.Image) -> torch.Tensor:
    return await asyncio.to_thread(get_embedding, image)