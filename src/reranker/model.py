from functools import lru_cache


RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def get_reranker(model_name=RERANKER_MODEL):
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(model_name, device=device, max_length=512)
