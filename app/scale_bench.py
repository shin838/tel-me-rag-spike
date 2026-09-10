"""규모별 부하 비교: pgvector vs Qdrant.

  docker compose exec api python -m app.scale_bench                 # 1000, 10000
  docker compose exec api python -m app.scale_bench --sizes 1000 10000 50000

임베딩 없이 합성 벡터를 쓴다. 목적은 검색 품질이 아니라 '규모가 커질 때
자원과 지연이 어떻게 변하는가' 이므로 벡터의 의미는 필요 없다.
측정 후 생성한 더미 데이터는 스스로 정리한다.
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from . import config
from .stores import get_store
from .stores.pgvector_store import connect

DIM = 1024
PREFIX = "SCALE-"


def seed_dummy_faqs(n: int) -> list[str]:
    """pgvector 테이블이 faq(id) 를 참조하므로 더미 원본을 먼저 만든다."""
    ids = [f"{PREFIX}{i:06d}" for i in range(n)]
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO faq (id, category, question, answer)
               VALUES (%s, '규모테스트', %s, '더미')
               ON CONFLICT (id) DO NOTHING""",
            [(i, f"규모 테스트용 더미 질문 {i}") for i in ids],
        )
    return ids


def cleanup() -> None:
    with connect() as conn:
        conn.execute(f"DELETE FROM faq WHERE id LIKE '{PREFIX}%'")


def pg_table_bytes(model_key: str) -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT pg_total_relation_size(%s)", (f"faq_vec_{model_key}",))
        return int(cur.fetchone()[0])


def qdrant_disk_bytes(model_key: str) -> int:
    from .stores.qdrant_store import client, _collection
    try:
        info = client().get_collection(_collection(model_key))
        return int(getattr(info, "indexed_vectors_count", 0) or 0)  # 참고용
    except Exception:
        return 0


def bench(store_name: str, model_key: str, ids: list[str], vectors: np.ndarray,
          queries: np.ndarray) -> dict:
    store = get_store(store_name)
    store.ensure(model_key, DIM)

    t0 = time.perf_counter()
    batch = 1000
    for i in range(0, len(ids), batch):
        store.upsert(model_key, ids[i:i + batch], vectors[i:i + batch].tolist())
    upsert_ms = (time.perf_counter() - t0) * 1000

    index_ms = store.build_index(model_key)

    latencies = []
    for q in queries:
        _, ms = store.search(model_key, q.tolist(), 3)
        latencies.append(ms)
    latencies.sort()

    return {
        "store": store_name,
        "count": store.count(model_key),
        "upsert_ms": upsert_ms,
        "index_ms": index_ms,
        "search_avg": statistics.mean(latencies),
        "search_p95": latencies[max(0, int(len(latencies) * 0.95) - 1)],
        "search_max": latencies[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="규모별 pgvector vs Qdrant 비교")
    ap.add_argument("--sizes", nargs="*", type=int, default=[1000, 10000])
    ap.add_argument("--queries", type=int, default=50)
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    print(f"차원 {DIM}, 질의 {args.queries}회, 합성 벡터\n")

    rows = []
    try:
        for n in args.sizes:
            key = f"scale{n}"
            print(f"=== {n:,}건 ===")
            ids = seed_dummy_faqs(n)
            vectors = rng.normal(size=(n, DIM)).astype(np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            queries = rng.normal(size=(args.queries, DIM)).astype(np.float32)
            queries /= np.linalg.norm(queries, axis=1, keepdims=True)
            print(f"  원본 벡터 크기 {vectors.nbytes / 1024 / 1024:.1f}MB")

            for store_name in config.STORES:
                r = bench(store_name, key, ids, vectors, queries)
                r["n"] = n
                if store_name == "pgvector":
                    r["disk_mb"] = pg_table_bytes(key) / 1024 / 1024
                rows.append(r)
                disk = f", 테이블 {r['disk_mb']:.1f}MB" if "disk_mb" in r else ""
                print(f"  [{store_name:9s}] 적재 {r['upsert_ms']/1000:5.1f}s, "
                      f"인덱스 {r['index_ms']/1000:5.1f}s, "
                      f"검색 평균 {r['search_avg']:6.1f}ms / p95 {r['search_p95']:6.1f}ms{disk}")
            print()
    finally:
        cleanup()
        print("더미 데이터 정리 완료")

    print("\n| 건수 | 스토어 | 적재 | 인덱스 | 검색 평균 | 검색 p95 |")
    print("| ---: | --- | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(f"| {r['n']:,} | {r['store']} | {r['upsert_ms']/1000:.1f}s | "
              f"{r['index_ms']/1000:.1f}s | {r['search_avg']:.1f}ms | {r['search_p95']:.1f}ms |")


if __name__ == "__main__":
    main()
