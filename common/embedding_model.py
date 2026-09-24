from functools import lru_cache


MODEL_NAME = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def get_model(model_name=MODEL_NAME):
    # Heavy imports are deferred so query-side modules can be imported (and
    # tested) without the ML stack installed.
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def embed_texts(texts, batch_size=32, model_name=MODEL_NAME):
    model = get_model(model_name)

    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


SPARSE_MODEL_NAME = "Qdrant/bm25"


@lru_cache(maxsize=1)
def get_sparse_model(model_name=SPARSE_MODEL_NAME):
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=model_name)


def embed_sparse(texts, model_name=SPARSE_MODEL_NAME):
    return list(get_sparse_model(model_name).embed(texts))


def embed_sparse_query(text, model_name=SPARSE_MODEL_NAME):
    return list(get_sparse_model(model_name).query_embed(text))[0]
