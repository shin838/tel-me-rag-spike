"""유사도 임계값 스윕 (FR-B7).

이용재 스파이크의 SIM_THRESHOLD=0.45 는 '답 없는 질문'으로 검증된 값이 아니다
(EXPERIMENT.md 6.4: "답 있는 질문 중 최저점보다 아래로만 잡았다").

이 스크립트는:
  - 답 있는 질문 30건 (eval_questions.json)
  - 답 없는 질문 15건 (eval_out_of_scope.json)
각각의 top-1 유사도 분포를 구하고, 임계값을 훑으며
  답있음 통과율 (높아야 좋음) vs 답없음 거부율 (높아야 좋음)
을 표로 만들어 제안 임계값을 낸다.

  python -m app.threshold_eval
  python -m app.threshold_eval --model bge-m3 --store pgvector --k 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime

from . import config, rag


def load_json(name: str) -> list[dict]:
    with open(os.path.join(config.DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def collect_scores(questions: list[dict], *, model: str, store: str, k: int, has_gold: bool) -> list[dict]:
    """각 질문을 실제 파이프라인으로 검색해 top-1 유사도를 기록한다.
    답 있는 질문은 정답 FAQ가 top-k 안에 들어왔는지도 함께 본다."""
    out = []
    for q in questions:
        r = rag.retrieve(q["query"], embed_model=model, store_name=store, k=k)
        got = [s.faq_id for s in r.sources]
        gold_hit = bool(has_gold and set(q.get("gold", [])) & set(got))
        out.append({
            "id": q["id"],
            "type": q.get("type", ""),
            "query": q["query"],
            "top1": round(r.top_score, 4),
            "gold_hit": gold_hit,
        })
    return out


def summarize(scores: list[dict]) -> dict:
    vals = [s["top1"] for s in scores]
    return {"min": min(vals), "median": statistics.median(vals),
            "mean": statistics.mean(vals), "max": max(vals)}


def sweep(answerable: list[dict], oos: list[dict], thresholds: list[float]) -> list[dict]:
    n_ans, n_oos = len(answerable), len(oos)
    rows = []
    for t in thresholds:
        passed = [s for s in answerable if s["top1"] >= t]          # 통과 = LLM 호출
        useful = [s for s in passed if s["gold_hit"]]               # 통과 + 정답이 실제로 top-k에 있음
        rejected = [s for s in oos if s["top1"] < t]                # 거부 = "확인 불가" 안내 (원하는 동작)
        rows.append({
            "threshold": t,
            "answerable_pass_rate": len(passed) / n_ans,
            "answerable_useful_rate": len(useful) / n_ans,
            "oos_reject_rate": len(rejected) / n_oos,
        })
    return rows


def recommend(rows: list[dict], min_pass: float = 0.95) -> dict | None:
    """답 있는 질문 통과율이 min_pass 이상인 것 중 거부율이 가장 높은(동률이면 더 높은) 임계값."""
    ok = [r for r in rows if r["answerable_pass_rate"] >= min_pass]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["oos_reject_rate"], r["threshold"]))


def to_markdown(model, store, k, n_answerable, ans_stat, oos_stat, oos_scores, rows, rec, min_pass) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [
        f"# 유사도 임계값 스윕 (FR-B7) — {ts}", "",
        f"- 임베딩 `{model}` / Vector DB `{store}` / top-k={k}",
        f"- 답 있는 질문 {n_answerable}건 (`eval_questions.json`)",
        f"- 답 없는 질문 {len(oos_scores)}건 (`eval_out_of_scope.json`)",
        f"- 참고: 이용재 스파이크 기본값 `SIM_THRESHOLD={config.SIM_THRESHOLD}`", "",
        "## 1. top-1 유사도 분포", "",
        "| 그룹 | 최소 | 중앙값 | 평균 | 최대 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| 답 있는 질문 | {ans_stat['min']:.3f} | {ans_stat['median']:.3f} | {ans_stat['mean']:.3f} | {ans_stat['max']:.3f} |",
        f"| 답 없는 질문 | {oos_stat['min']:.3f} | {oos_stat['median']:.3f} | {oos_stat['mean']:.3f} | {oos_stat['max']:.3f} |",
        "",
        "> 두 분포가 거의 안 겹치면(답없음 최대 < 답있음 최소) 그 사이 아무 값이나 좋은 임계값.",
        "> 크게 겹치면 벡터 검색만으로는 범위 밖 질문을 못 거른다 → 리랭커/의도 분류기 필요.", "",
        "## 2. 임계값 스윕", "",
        "| 임계값 | 답있음 통과율 | 답있음 정답통과율 | 답없음 거부율 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        mark = "  ← 제안" if rec and abs(r["threshold"] - rec["threshold"]) < 1e-9 else ""
        L.append(f"| {r['threshold']:.2f} | {r['answerable_pass_rate']:.1%} | "
                 f"{r['answerable_useful_rate']:.1%} | {r['oos_reject_rate']:.1%} |{mark}")
    L += ["", "## 3. 제안", ""]
    if rec:
        L.append(f"**임계값 {rec['threshold']:.2f}** — 답 있는 질문 {rec['answerable_pass_rate']:.0%} 통과 유지, "
                 f"답 없는 질문 {rec['oos_reject_rate']:.0%} 거부.")
        L.append("")
        L.append(f"(기준: 답있음 통과율 {min_pass:.0%} 이상을 유지하면서 답없음 거부율 최대)")
    else:
        L.append(f"답 있는 질문 통과율 {min_pass:.0%} 이상을 만족하는 임계값이 없음. "
                 "두 분포가 크게 겹친다는 뜻 → 벡터 단독으로는 거부 판정에 한계.")
    L += [
        "",
        "> 주의: FAQ 100건 기준. 1,000건으로 확장하면 점수 분포가 달라지므로 재측정 필요",
        "> (이용재 EXPERIMENT.md 6.4와 동일한 한계).",
        "",
        "## 4. 답 없는 질문별 top-1 유사도 (높은 순)", "",
        "| ID | type | 질문 | top-1 |",
        "| --- | --- | --- | ---: |",
    ]
    for s in sorted(oos_scores, key=lambda x: -x["top1"]):
        L.append(f"| {s['id']} | {s['type']} | {s['query']} | {s['top1']:.3f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="유사도 임계값 스윕 (FR-B7)")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--store", default="pgvector")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--lo", type=float, default=0.30)
    ap.add_argument("--hi", type=float, default=0.60)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--min-pass", type=float, default=0.95, help="제안 임계값의 최소 통과율 기준")
    args = ap.parse_args()

    answerable = load_json("eval_questions.json")
    oos = load_json("eval_out_of_scope.json")
    print(f"답 있는 질문 {len(answerable)}건 / 답 없는 질문 {len(oos)}건")
    print(f"모델={args.model}, 스토어={args.store}, top-k={args.k}\n")

    ans_scores = collect_scores(answerable, model=args.model, store=args.store, k=args.k, has_gold=True)
    oos_scores = collect_scores(oos, model=args.model, store=args.store, k=args.k, has_gold=False)
    ans_stat, oos_stat = summarize(ans_scores), summarize(oos_scores)

    print(f"답 있는 질문 top-1: 최소 {ans_stat['min']:.3f} / 중앙 {ans_stat['median']:.3f} / 최대 {ans_stat['max']:.3f}")
    print(f"답 없는 질문 top-1: 최소 {oos_stat['min']:.3f} / 중앙 {oos_stat['median']:.3f} / 최대 {oos_stat['max']:.3f}\n")

    n = int(round((args.hi - args.lo) / args.step)) + 1
    thresholds = [round(args.lo + i * args.step, 2) for i in range(n)]
    rows = sweep(ans_scores, oos_scores, thresholds)
    rec = recommend(rows, args.min_pass)

    print(f"{'임계값':>6}  {'답있음통과':>10}  {'정답통과':>8}  {'답없음거부':>10}")
    for r in rows:
        print(f"{r['threshold']:>6.2f}  {r['answerable_pass_rate']:>10.1%}  "
              f"{r['answerable_useful_rate']:>8.1%}  {r['oos_reject_rate']:>10.1%}")
    print(f"\n제안 임계값: {rec['threshold']:.2f}" if rec else "\n제안 임계값 없음 (두 분포가 크게 겹침)")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    md = to_markdown(args.model, args.store, args.k, len(answerable), ans_stat, oos_stat,
                     oos_scores, rows, rec, args.min_pass)
    md_path = os.path.join(config.RESULTS_DIR, "threshold_sweep.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(config.RESULTS_DIR, "threshold_sweep.json"), "w", encoding="utf-8") as f:
        json.dump({"answerable": ans_scores, "oos": oos_scores, "sweep": rows, "recommended": rec},
                  f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {md_path}")


if __name__ == "__main__":
    main()
