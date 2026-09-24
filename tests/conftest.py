import types

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models


TEXT_A1 = "ರಾಜಪುರೋಹಿತರು 1880 ರಲ್ಲಿ ಧಾರವಾಡದಲ್ಲಿ ಜನಿಸಿದರು ಎಂದು ತಿಳಿದುಬರುತ್ತದೆ"
TEXT_A2 = "ಕುವೆಂಪು ಕುಪ್ಪಳ್ಳಿಯಲ್ಲಿ ಹುಟ್ಟಿ ಕವಿತೆಗಳನ್ನು ಬರೆದರು ಮತ್ತು ನಾಟಕ ರಚಿಸಿದರು"
TEXT_B1 = "ರಾಜಪುರೋಹಿತರು ಧಾರವಾಡದಲ್ಲಿ 1880 ರಲ್ಲಿ ಜನಿಸಿದರು ಎಂದು ಹೇಳಲಾಗಿದೆ"


def make_chunk(chunk_id, book_id, text, page=1, title=None, author="A"):
    return {
        "chunk_id": chunk_id, "book_id": book_id, "title": title or f"Book {book_id}",
        "author": author, "language": "kn", "page_start": page, "page_end": page,
        "paragraph_start": 1, "paragraph_end": 2, "text": text, "char_count": len(text),
    }


@pytest.fixture
def qdrant():
    """In-memory Qdrant with the production collection layout (dense cosine +
    IDF sparse) and 3 hand-made points. Dense dim 3; sparse vocab ids 0..4."""
    client = QdrantClient(":memory:")
    client.create_collection(
        "kannada_chunks",
        vectors_config={"dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    rows = [
        (make_chunk("A_0001", "A", TEXT_A1, page=3), [1, 0, 0], ([0], [1.0])),
        (make_chunk("A_0002", "A", TEXT_A2, page=4), [0, 1, 0], ([1], [1.0])),
        (make_chunk("B_0001", "B", TEXT_B1, page=9), [0.9, 0.1, 0], ([0, 2], [1.0, 1.0])),
    ]
    client.upsert("kannada_chunks", [
        models.PointStruct(
            id=i + 1,
            vector={"dense": dense, "sparse": models.SparseVector(indices=sp[0], values=sp[1])},
            payload=chunk,
        )
        for i, (chunk, dense, sp) in enumerate(rows)
    ])
    return client


def sparse(indices, values=None):
    return types.SimpleNamespace(indices=indices, values=values or [1.0] * len(indices))
