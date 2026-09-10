"""Qdrant 구현.

FAQ 원본은 PostgreSQL 에 두고 벡터만 Qdrant 에 둔다(= 본 프로젝트의 'PostgreSQL + Qdrant' 구성).
그래서 FR-D3 동기화는 트랜잭션이 아니라 애플리케이션 책임이 된다 - 이 차이가 비교의 핵심이다.
"""
from __future__ import annotations

import re
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .. import config
from .base import Hit, VectorStore

_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    return _client


def _collection(model_key: str) -> str:
    return f"faq_{model_key}"


def _point_id(faq_id: str) -> int:
    """'FAQ-001' -> 1. Qdrant 포인트 ID 는 정수 또는 UUID 만 허용한다."""
    m = re.search(r"(\d+)", faq_id)
    if not m:
        raise ValueError(f"숫자를 포함하지 않는 FAQ id: {faq_id}")
    return int(m.group(1))


class QdrantStore(VectorStore):
    name = "qdrant"

    def ensure(self, model_key: str, dim: int) -> None:
        c = _collection(model_key)
        client().recreate_collection(
            collection_name=c,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=64),
        )

    def upsert(self, model_key: str, faq_ids: list[str], vectors: list[list[float]]) -> None:
        points = [
            qm.PointStruct(id=_point_id(fid), vector=vec, payload={"faq_id": fid})
            for fid, vec in zip(faq_ids, vectors)
        ]
        client().upsert(collection_name=_collection(model_key), points=points, wait=True)

    def build_index(self, model_key: str) -> float:
        """Qdrant 는 색인을 자동으로 만든다. 최적화가 끝날 때까지 기다린 시간을 잰다."""
        c = _collection(model_key)
        started = time.perf_counter()
        deadline = started + 60
        while time.perf_counter() < deadline:
            info = client().get_collection(c)
            if info.status == qm.CollectionStatus.GREEN:
                break
            time.sleep(0.2)
        return (time.perf_counter() - started) * 1000

    def search(self, model_key: str, vector: list[float], k: int) -> tuple[list[Hit], float]:
        started = time.perf_counter()
        res = client().query_points(
            collection_name=_collection(model_key),
            query=vector,
            limit=k,
            with_payload=True,
        ).points
        hits = [Hit(faq_id=p.payload["faq_id"], score=float(p.score)) for p in res]
        return hits, (time.perf_counter() - started) * 1000

    def count(self, model_key: str) -> int:
        try:
            return int(client().count(_collection(model_key), exact=True).count)
        except Exception:
            return 0
