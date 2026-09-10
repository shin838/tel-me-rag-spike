"""임베딩 3종 x Vector DB 2종 비교 실험.

  python -m app.evaluate            # 6개 조합 전부
  python -m app.evaluate --k 5

질의 임베딩은 모델당 한 번만 계산해 두 스토어가 같은 벡터로 검색한다.
그래야 지표 차이가 'DB 차이'에서 온 것인지 '임베딩 차이'에서 온 것인지 섞이지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime

from . import config, ollama_client
from .stores import get_store
from .stores.pgvector_store import connect


def load_eval() -> list[dict]:
    with open(os.path.join(config.DATA_DIR, "eval_questions.json"), encoding="utf-8") as f:
        return json.load(f)


def embed_queries(questions: list[dict], model_name: str) -> tuple[list[list[float]], float]:
    vectors, per_call = [], []
    for q in questions:
        vec, ms = ollama_client.embed_one(q["query"], model_name, is_query=True)
        vectors.append(vec)
        per_call.append(ms)
    return vectors, statistics.mean(per_call)


def evaluate_combo(model_name: str, store_name: str, questions: list[dict],
                   query_vectors: list[list[float]], k: int) -> dict:
    key = config.embed_conf(model_name)["key"]
    store = get_store(store_name)

    if store.count(key) == 0:
        raise RuntimeError(f"{store_name}/{model_name} 에 적재된 벡터가 없습니다. 먼저 python -m app.ingest 를 실행하세요.")

    latencies: list[float] = []
    hit1 = hit3 = hit5 = 0
    rr_total = 0.0
    misses = []

    for q, vec in zip(questions, query_vectors):
        hits, ms = store.search(key, vec, k)
        latencies.append(ms)
        ranked = [h.faq_id for h in hits]
        gold = set(q["gold"])

        rank = next((i for i, fid in enumerate(ranked, start=1) if fid in gold), None)
        if rank is not None:
            rr_total += 1.0 / rank
            if rank <= 1: hit1 += 1
            if rank <= 3: hit3 += 1
            if rank <= 5: hit5 += 1
        else:
            misses.append({"id": q["id"], "query": q["query"], "gold": q["gold"],
                           "got": ranked[:3], "top_score": round(hits[0].score, 4) if hits else 0.0})

    n = len(questions)
    latencies_sorted = sorted(latencies)
    p95 = latencies_sorted[max(0, int(len(latencies_sorted) * 0.95) - 1)]
    return {
        "embed_model": model_name, "store": store_name, "n": n, "k": k,
        "recall@1": hit1 / n, "recall@3": hit3 / n, "recall@5": hit5 / n,
        "mrr@5": rr_total / n,
        "search_ms_avg": statistics.mean(latencies), "search_ms_p95": p95,
        "misses": misses,
    }


def save_to_db(rows: list[dict], dims: dict[str, int], embed_ms: dict[str, float]) -> None:
    with connect() as conn, conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO experiment_run
                   (embed_model, store, dim, recall_at_1, recall_at_3, recall_at_5, mrr_at_5,
                    search_ms_avg, search_ms_p95, embed_ms_avg, notes)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (r["embed_model"], r["store"], dims[r["embed_model"]],
                 r["recall@1"], r["recall@3"], r["recall@5"], r["mrr@5"],
                 r["search_ms_avg"], r["search_ms_p95"], embed_ms[r["embed_model"]],
                 f"평가질문 {r['n']}건, top-k={r['k']}"),
            )


def to_markdown(rows: list[dict], dims: dict[str, int], embed_ms: dict[str, float], k: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 임베딩 x Vector DB 비교 결과 ({ts})", "",
        f"- 평가 질문 {rows[0]['n']}건, top-k={k}, FAQ 100건",
        f"- LLM: `{config.LLM_MODEL}` (검색 지표에는 영향 없음)", "",
        "## 종합", "",
        "| 임베딩 | 차원 | Vector DB | Recall@1 | Recall@3 | Recall@5 | MRR@5 | 검색 평균 | 검색 p95 | 질의 임베딩 평균 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        m = r["embed_model"]
        lines.append(
            f"| {m} | {dims[m]} | {r['store']} | {r['recall@1']:.1%} | {r['recall@3']:.1%} | "
            f"{r['recall@5']:.1%} | {r['mrr@5']:.3f} | {r['search_ms_avg']:.1f}ms | "
            f"{r['search_ms_p95']:.1f}ms | {embed_ms[m]:.0f}ms |"
        )

    lines += ["", "## 검색 실패 사례 (top-3 안에 정답 FAQ 없음)", ""]
    for r in rows:
        if not r["misses"]:
            lines.append(f"### {r['embed_model']} / {r['store']} — 실패 없음")
            continue
        lines.append(f"### {r['embed_model']} / {r['store']} — {len(r['misses'])}건")
        lines.append("")
        lines.append("| 질문 ID | 질문 | 정답 | 검색 결과 top-3 | 최고 점수 |")
        lines.append("| --- | --- | --- | --- | ---: |")
        for m in r["misses"]:
            lines.append(f"| {m['id']} | {m['query']} | {', '.join(m['gold'])} | "
                         f"{', '.join(m['got'])} | {m['top_score']:.3f} |")
        lines.append("")

    lines += [
        "", "## 읽는 법", "",
        "- **Recall@3**: NFR-3.2 목표는 80% 이상. 이 수치가 임베딩 모델 선택의 1순위 기준이다.",
        "- **검색 지연**: NFR-1.1 목표는 500ms 이내. FAQ 100건 규모에서는 두 DB 모두 여유가 크므로,",
        "  지연만으로 DB를 고르지 말고 운영 복잡도(FR-D3 동기화)를 함께 보라.",
        "- **질의 임베딩 평균**: CPU 환경에서는 이 값이 검색 지연보다 10~100배 크다. 체감 속도의 실제 병목이다.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="임베딩 x Vector DB 비교 실험")
    ap.add_argument("--models", nargs="*", default=list(config.EMBED_MODELS))
    ap.add_argument("--stores", nargs="*", default=config.STORES)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    questions = load_eval()
    print(f"평가 질문 {len(questions)}건 로드, top-k={args.k}")

    rows, dims, embed_ms = [], {}, {}
    for model_name in args.models:
        print(f"\n=== {model_name} 질의 임베딩 ===")
        started = time.perf_counter()
        vectors, avg_ms = embed_queries(questions, model_name)
        dims[model_name] = len(vectors[0])
        embed_ms[model_name] = avg_ms
        print(f"  {len(vectors)}건, {dims[model_name]}차원, 건당 평균 {avg_ms:.0f}ms "
              f"(총 {(time.perf_counter()-started):.1f}초)")

        for store_name in args.stores:
            r = evaluate_combo(model_name, store_name, questions, vectors, args.k)
            rows.append(r)
            print(f"  [{store_name:9s}] Recall@1 {r['recall@1']:.1%}  Recall@3 {r['recall@3']:.1%}  "
                  f"MRR@5 {r['mrr@5']:.3f}  검색 {r['search_ms_avg']:.1f}ms")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    md = to_markdown(rows, dims, embed_ms, args.k)
    md_path = os.path.join(config.RESULTS_DIR, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(config.RESULTS_DIR, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "dims": dims, "query_embed_ms_avg": embed_ms}, f,
                  ensure_ascii=False, indent=2)
    try:
        save_to_db(rows, dims, embed_ms)
    except Exception as e:
        print(f"  (experiment_run 테이블 기록 실패: {e})")

    print(f"\n결과 저장: {md_path}")
    print("\n" + md.split("## 검색 실패")[0])


if __name__ == "__main__":
    main()
