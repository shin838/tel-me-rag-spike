"""FAQ 적재 파이프라인.

  python -m app.ingest                       # 전체 모델 x 전체 스토어
  python -m app.ingest --models bge-m3       # 특정 임베딩만
  python -m app.ingest --stores qdrant

임베딩은 모델당 한 번만 계산해 results/ 에 캐시하고 두 스토어가 같은 벡터를 공유한다.
같은 벡터를 넣어야 'DB 차이'만 비교할 수 있고, CPU 임베딩 시간도 아낄 수 있다.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import config, ollama_client
from .stores import get_store
from .stores.pgvector_store import connect


def load_faqs() -> list[dict]:
    with open(os.path.join(config.DATA_DIR, "faqs.json"), encoding="utf-8") as f:
        return json.load(f)


def doc_text(faq: dict) -> str:
    """임베딩 대상 텍스트. 질문만 넣는 것보다 답변을 함께 넣는 편이 재현율이 높다."""
    return f"[{faq['category']}] {faq['question']}\n{faq['answer']}"


def load_faqs_into_rdb(faqs: list[dict]) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO faq (id, category, question, answer, keywords)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE
                 SET category = EXCLUDED.category,
                     question = EXCLUDED.question,
                     answer   = EXCLUDED.answer,
                     keywords = EXCLUDED.keywords""",
            [(f["id"], f["category"], f["question"], f["answer"], f.get("keywords", [])) for f in faqs],
        )
    print(f"  RDB(faq 테이블) 적재 완료: {len(faqs)}건")


def cache_path(model_name: str) -> str:
    key = config.embed_conf(model_name)["key"]
    return os.path.join(config.RESULTS_DIR, f"embeddings_{key}.npz")


def embed_documents(faqs: list[dict], model_name: str, *, use_cache: bool = True) -> tuple[np.ndarray, list[str], float]:
    path = cache_path(model_name)
    ids = [f["id"] for f in faqs]
    if use_cache and os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        if list(data["ids"]) == ids:
            print(f"  임베딩 캐시 사용: {path}")
            return data["vectors"], ids, float(data["embed_ms"])
        print("  캐시의 FAQ 목록이 달라 다시 임베딩합니다.")

    texts = [doc_text(f) for f in faqs]
    print(f"  {model_name} 으로 {len(texts)}건 임베딩 중... (CPU 환경에서는 수 분 걸릴 수 있습니다)")
    started = time.perf_counter()
    vectors: list[list[float]] = []
    batch = 8  # CPU 메모리 보호용 소배치
    for i in range(0, len(texts), batch):
        vecs, _ = ollama_client.embed(texts[i:i + batch], model_name, is_query=False)
        vectors.extend(vecs)
        print(f"    {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    total_ms = (time.perf_counter() - started) * 1000
    arr = np.asarray(vectors, dtype=np.float32)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    np.savez(path, ids=np.array(ids), vectors=arr, embed_ms=total_ms)
    print(f"\n  임베딩 완료: {arr.shape[0]}건 x {arr.shape[1]}차원, {total_ms/1000:.1f}초 "
          f"(건당 {total_ms/len(texts):.0f}ms)")
    return arr, ids, total_ms


def report_duplicates(vectors: np.ndarray, faqs: list[dict], threshold: float = 0.95) -> list[tuple[str, str, float]]:
    """NFR-3.3: 유사도 0.95 이상인 FAQ 쌍을 찾아 보고한다(삭제는 사람이 판단)."""
    norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    sim = norm @ norm.T
    np.fill_diagonal(sim, 0.0)
    pairs = []
    for i, j in zip(*np.where(sim >= threshold)):
        if i < j:
            pairs.append((faqs[i]["id"], faqs[j]["id"], float(sim[i, j])))
    return sorted(pairs, key=lambda p: -p[2])


def ingest(model_name: str, store_name: str, faqs: list[dict], vectors: np.ndarray, ids: list[str]) -> dict:
    key = config.embed_conf(model_name)["key"]
    store = get_store(store_name)
    dim = int(vectors.shape[1])

    store.ensure(key, dim)
    started = time.perf_counter()
    store.upsert(key, ids, vectors.tolist())
    upsert_ms = (time.perf_counter() - started) * 1000
    index_ms = store.build_index(key)
    count = store.count(key)

    print(f"  [{store_name} / {model_name}] {count}건 적재, "
          f"upsert {upsert_ms:.0f}ms, 인덱스 {index_ms:.0f}ms, {dim}차원")
    return {"store": store_name, "model": model_name, "dim": dim,
            "count": count, "upsert_ms": upsert_ms, "index_ms": index_ms}


def main() -> None:
    ap = argparse.ArgumentParser(description="FAQ 를 RDB 와 Vector DB 에 적재한다")
    ap.add_argument("--models", nargs="*", default=list(config.EMBED_MODELS))
    ap.add_argument("--stores", nargs="*", default=config.STORES)
    ap.add_argument("--no-cache", action="store_true", help="임베딩 캐시를 무시하고 다시 계산")
    args = ap.parse_args()

    faqs = load_faqs()
    print(f"FAQ {len(faqs)}건 로드")
    load_faqs_into_rdb(faqs)

    summary = []
    for model_name in args.models:
        print(f"\n=== 임베딩 모델: {model_name} ===")
        vectors, ids, embed_ms = embed_documents(faqs, model_name, use_cache=not args.no_cache)

        dups = report_duplicates(vectors, faqs)
        if dups:
            print(f"  [주의] 유사도 0.95 이상 중복 후보 {len(dups)}쌍:")
            for a, b, s in dups[:10]:
                print(f"    {a} <-> {b} : {s:.3f}")
        else:
            print("  중복 후보(유사도 0.95 이상) 없음")

        for store_name in args.stores:
            summary.append(ingest(model_name, store_name, faqs, vectors, ids))

    print("\n적재 요약")
    for s in summary:
        print(f"  {s['model']:24s} {s['store']:9s} {s['count']:4d}건 {s['dim']:5d}차원")


if __name__ == "__main__":
    main()
