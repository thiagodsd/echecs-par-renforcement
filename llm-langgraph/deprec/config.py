# config.py
import os
from dotenv import load_dotenv

load_dotenv("credentials.env", override=True)

CONFIG = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
        },
    },
    "chunker": {
        "chunk_size": 1024,
        "chunk_overlap": 100,
        "min_chunk_size": 200,
    },
}