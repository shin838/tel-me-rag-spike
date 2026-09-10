"""Ollama 없이 DB 배선만 검증한다(임의 벡터 사용).

  docker compose run --rm --no-deps api python -m app.smoke_test

모델 내려받기 전에 pgvector/Qdrant 구현과 SQL 이 맞는지 먼저 확인하기 위한 용도다.
"""
from __future__ import annotations

import sys

import numpy as np

from . import config
from .ingest import load_faqs, load_faqs_into_rdb
from .stores import get_store

DIM = 1024
KEY = "smoke"


def main() -> int:
    rng = np.random.default_rng(42)
    faqs = load_faqs()
    ids = [f["id"] for f in faqs]

    print(f"FAQ {len(faqs)}건 -> RDB 적재")
    load_faqs_into_rdb(faqs)

    vectors = rng.normal(size=(len(ids), DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    failed = False
    for store_name in config.STORES:
        store = get_store(store_name)
        print(f"\n=== {store_name} ===")
        store.ensure(KEY, DIM)
        store.upsert(KEY, ids, vectors.tolist())
        index_ms = store.build_index(KEY)
        count = store.count(KEY)
        print(f"  적재 {count}건, 인덱스 {index_ms:.0f}ms")

        # 문서 벡터 자신으로 검색하면 자기 자신이 1위여야 한다.
        ok = 0
        latencies = []
        for i in (0, 17, 42, 63, 99):
            hits, ms = store.search(KEY, vectors[i].tolist(), 3)
            latencies.append(ms)
            if hits and hits[0].faq_id == ids[i] and hits[0].score > 0.99:
                ok += 1
            else:
                print(f"  [실패] {ids[i]} -> {[(h.faq_id, round(h.score,3)) for h in hits[:2]]}")
        print(f"  자기검색 정확도 {ok}/5, 검색 평균 {np.mean(latencies):.1f}ms")

        if count != len(ids) or ok != 5:
            failed = True

    print("\n" + ("실패한 항목이 있습니다." if failed else "두 스토어 모두 정상입니다."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
