from .base import Hit, VectorStore
from .pgvector_store import PgVectorStore
from .qdrant_store import QdrantStore

_REGISTRY: dict[str, VectorStore] = {}


def get_store(name: str) -> VectorStore:
    """설정 문자열 하나로 스토어를 교체한다."""
    if name not in _REGISTRY:
        if name == "pgvector":
            _REGISTRY[name] = PgVectorStore()
        elif name == "qdrant":
            _REGISTRY[name] = QdrantStore()
        else:
            raise ValueError(f"알 수 없는 스토어: {name} (가능: pgvector, qdrant)")
    return _REGISTRY[name]


__all__ = ["Hit", "VectorStore", "PgVectorStore", "QdrantStore", "get_store"]
