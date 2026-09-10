"""CPU 환경에서 LLM 응답 시간을 측정한다(NFR-1.3 첫 토큰 3초, NFR-1.4 전체 15초).

  docker compose exec api python -m app.llm_bench
  docker compose exec api python -m app.llm_bench --model exaone3.5:2.4b

이 수치가 Structure.md 6.4 '로컬 LLM 운영 방식 A/B/C' 결정의 근거가 된다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics

from . import config, rag

QUESTIONS = [
    "제일 싼 요금제 뭐예요?",
    "약정 1년쯤 지났는데 지금 해지하면 얼마 물어내요?",
    "일본 가는데 데이터 어떻게 써요?",
    "애들 이름으로 폰 만들려면 뭐 챙겨가요?",
    "해지 매장 안 가고 집에서 되나요?",
]


async def run_one(question: str, model: str, embed_model: str, store: str) -> dict:
    answer, retrieval, done, error = "", None, None, None
    async for ev in rag.answer_stream(question, embed_model=embed_model, store_name=store, llm_model=model):
        if ev["type"] == "retrieval":
            retrieval = ev
        elif ev["type"] == "token":
            answer += ev["text"]
        elif ev["type"] == "done":
            done = ev
        elif ev["type"] == "error":
            error = ev

    return {
        "question": question,
        "answer": answer.strip(),
        "error": error["message"] if error else None,
        "embed_ms": retrieval["embed_ms"] if retrieval else None,
        "search_ms": retrieval["search_ms"] if retrieval else None,
        "top_score": retrieval["top_score"] if retrieval else None,
        "sources": [s["faq_id"] for s in retrieval["sources"]] if retrieval else [],
        "first_token_ms": done["first_token_ms"] if done else None,
        "total_ms": done["total_ms"] if done else None,
    }


async def main_async(args) -> None:
    model = args.model or config.LLM_MODEL
    print(f"LLM: {model} | 임베딩: {args.embed_model} | 스토어: {args.store}")

    if not args.no_warmup:
        # 첫 호출에는 모델 로딩 시간(CPU에서 10초 이상)이 섞인다. 버리는 호출로 분리한다.
        print("워밍업 중(모델 로딩)...")
        await run_one("요금제 알려주세요", model, args.embed_model, args.store)
    print()

    results = []
    for q in QUESTIONS:
        r = await run_one(q, model, args.embed_model, args.store)
        results.append(r)
        if r["error"]:
            print(f"[실패] {q}\n  {r['error']}\n")
            continue
        print(f"Q. {q}")
        print(f"A. {r['answer'][:220]}{'…' if len(r['answer']) > 220 else ''}")
        print(f"   근거 {', '.join(r['sources'])} (최고 유사도 {r['top_score']})")
        print(f"   임베딩 {r['embed_ms']}ms · 검색 {r['search_ms']}ms · "
              f"첫 토큰 {r['first_token_ms']/1000:.1f}s · 전체 {r['total_ms']/1000:.1f}s\n")

    ok = [r for r in results if not r["error"]]
    if not ok:
        print("측정 가능한 결과가 없습니다.")
        return

    ft = [r["first_token_ms"] for r in ok]
    tt = [r["total_ms"] for r in ok]
    print("=" * 60)
    print(f"첫 토큰  평균 {statistics.mean(ft)/1000:5.1f}s  최대 {max(ft)/1000:5.1f}s   "
          f"NFR-1.3(3초) {'충족' if max(ft) <= 3000 else '미달'}")
    print(f"전체     평균 {statistics.mean(tt)/1000:5.1f}s  최대 {max(tt)/1000:5.1f}s   "
          f"NFR-1.4(15초) {'충족' if max(tt) <= 15000 else '미달'}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"llm_bench_{model.replace(':', '_').replace('/', '_')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "embed_model": args.embed_model, "store": args.store,
                   "results": results,
                   "first_token_ms_avg": statistics.mean(ft), "first_token_ms_max": max(ft),
                   "total_ms_avg": statistics.mean(tt), "total_ms_max": max(tt)},
                  f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU LLM 응답 시간 측정")
    ap.add_argument("--model", default=None)
    ap.add_argument("--embed-model", dest="embed_model", default=config.DEFAULT_EMBED_MODEL)
    ap.add_argument("--store", default=config.DEFAULT_STORE)
    ap.add_argument("--no-warmup", dest="no_warmup", action="store_true",
                    help="워밍업 호출 없이 측정(콜드 스타트 시간을 보고 싶을 때)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
