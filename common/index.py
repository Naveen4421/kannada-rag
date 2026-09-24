"""Qdrant connection and collection name shared by ingestion (writes) and
retrieval (reads)."""
import os
from pathlib import Path

from qdrant_client import QdrantClient


INDEX_DIR = Path("data/index/qdrant")
COLLECTION_NAME = "kannada_chunks"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"


def get_client():
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=QDRANT_PREFER_GRPC,
    )
