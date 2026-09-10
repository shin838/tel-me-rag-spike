"""PostgreSQL + pgvector 구현.

모델마다 차원이 달라 테이블을 모델별로 분리한다(faq_vec_bge_m3 등).
FAQ 원본과 같은 DB 안에 있으므로 FR-D3(변경 시 임베딩 동기화)을 트랜잭션으로 묶을 수 있다.
"""
from __future__ import annotations

import time

import psycopg

from .. import config
from .base import Hit, VectorStore


def connect() -> psycopg.Connection:
    return psycopg.connect(**config.PG, autocommit=True)


def _table(model_key: str) -> str:
    return f"faq_vec_{model_key}"


def to_pgvector(vec: list[float]) -> str:
    """psycopg 가 vector 타입을 모르므로 문자열로 넘기고 ::vector 로 캐스팅한다."""
    return "[" + ",".join(f"{v:.7g}" for v in vec) + "]"


class PgVectorStore(VectorStore):
    name = "pgvector"

    def ensure(self, model_key: str, dim: int) -> None:
        t = _table(model_key)
        with connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
            conn.execute(
                f"""
                CREATE TABLE {t} (
                    faq_id    TEXT PRIMARY KEY REFERENCES faq(id) ON DELETE CASCADE,
                    embedding vector({dim}) NOT NULL
                )
                """
            )

    def upsert(self, model_key: str, faq_ids: list[str], vectors: list[list[float]]) -> None:
        t = _table(model_key)
        rows = [(fid, to_pgvector(v)) for fid, v in zip(faq_ids, vectors)]
        with connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {t} (faq_id, embedding) VALUES (%s, %s::vector)
                    ON CONFLICT (faq_id) DO UPDATE SET embedding = EXCLUDED.embedding""",
                rows,
            )

    def build_index(self, model_key: str) -> float:
        t = _table(model_key)
        started = time.perf_counter()
        with connect() as conn:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{t}_hnsw ON {t} "
                f"USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
            )
            conn.execute(f"ANALYZE {t}")
        return (time.perf_counter() - started) * 1000

    def search(self, model_key: str, vector: list[float], k: int) -> tuple[list[Hit], float]:
        t = _table(model_key)
        qv = to_pgvector(vector)
        started = time.perf_counter()
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""SELECT faq_id, 1 - (embedding <=> %s::vector) AS score
                    FROM {t}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s""",
                (qv, qv, k),
            )
            hits = [Hit(faq_id=r[0], score=float(r[1])) for r in cur.fetchall()]
        return hits, (time.perf_counter() - started) * 1000

    def count(self, model_key: str) -> int:
        t = _table(model_key)
        with connect() as conn, conn.cursor() as cur:
            try:
                cur.execute(f"SELECT count(*) FROM {t}")
            except psycopg.errors.UndefinedTable:
                return 0
            return int(cur.fetchone()[0])
