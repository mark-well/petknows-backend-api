import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
url:str = os.environ.get("SUPABASE_URL")
key:str = os.environ.get("SUPABASE_KEY")
supabase:Client = create_client(url,key)

def get_all_pets():
    response = (
        supabase.table("pets")
        .select("*")
        .execute()
    )

    return response

def match_pets(embedding: list[float], threshold: float, count: int, model_version: str) -> list[dict]:
    response = supabase.rpc(
        "match_pets",
        {
            "query_embedding": embedding,
            "match_threshold": threshold,
            "match_count": count,
            "required_model_version": model_version,
        },
    ).execute()
    return response.data