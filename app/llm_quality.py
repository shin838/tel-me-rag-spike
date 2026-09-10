"""LLM 답변 품질 비교 하네스.

같은 질문 + 같은 검색 결과(bge-m3/pgvector top-3)를 여러 LLM에 주고
답변을 나란히 덤프한다. 채점은 사람이 블라인드로 한다.

  python -m app.llm_quality                 # 기본 프롬프트(base)
  python -m app.llm_quality --prompt v2      # 수정 프롬프트

산출물 (프롬프트 변형 이름이 파일명에 붙는다):
  results/llm_quality_blind__base.md    채점용 (모델명 A/B/C 로 가림, 질문마다 셔플)
  results/llm_quality_key__base.json    정답 키 (A=exaone ...), 채점 후 공개
  results/llm_quality_named__base.md    참고용 (모델명 그대로)
  results/llm_quality_score__base.csv   채점 입력 템플릿

주의:
  - 임계값(FR-B7)을 적용하지 않는다. 유사도가 낮아도 LLM 을 호출해
    "근거 없을 때 지어내는지 vs 확인 불가로 답하는지"를 본다.
  - qwen3 계열은 <think> 블록을 뱉으므로 제거하고 저장한다.
  - 모델을 바깥 루프로 돌려(모델당 한 번만 로드) 콜드 스타트를 줄인다.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import string
from datetime import datetime

from . import config, ollama_client, prompts, rag

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
REJECT_HINTS = ("확인이 어렵", "확인이 힘들", "확인하기 어렵", "안내가 어렵",
                "제공하기 어렵", "1600-0100", "고객센터")


def load_questions() -> list[dict]:
    with open(os.path.join(config.DATA_DIR, "eval_quality.json"), encoding="utf-8") as f:
        return json.load(f)


async def generate(system: str, user: str, model: str) -> dict:
    parts: list[str] = []
    err: str | None = None
    meta: dict = {}
    async for ev in ollama_client.chat_stream(system, user, model=model):
        t = ev.get("type")
        if t == "token":
            parts.append(ev.get("text", ""))
        elif t == "error":
            err = ev.get("message", "알 수 없는 오류")
        elif t == "done":
            meta = {"first_token_ms": ev.get("first_token_ms"), "total_ms": ev.get("total_ms")}
    raw = "".join(parts)
    had_think = bool(THINK_RE.search(raw))
    text = THINK_RE.sub("", raw).strip()
    return {
        "model": model,
        "text": f"[오류] {err}" if err else text,
        "error": err,
        "rejected_phrasing": any(h in text for h in REJECT_HINTS),
        "had_think_block": had_think,
        "first_token_ms": meta.get("first_token_ms"),
        "total_ms": meta.get("total_ms"),
    }


def retrieve_all(questions: list[dict], *, embed_model: str, store: str, k: int) -> list[dict]:
    ctx = []
    for q in questions:
        r = rag.retrieve(q["query"], embed_model=embed_model, store_name=store, k=k)
        src_lines = [
            f"- {s.faq_id} ({s.category}) · 유사도 {s.score:.3f}\n"
            f"  Q: {s.question}\n  A: {s.answer}"
            for s in r.sources
        ]
        ctx.append({
            "q": q,
            "top_score": round(r.top_score, 4),
            "sources_md": "\n".join(src_lines),
            "user_prompt": rag.build_user_prompt(q["query"], r.sources),
        })
    return ctx


def blind_labels(n: int, rng: random.Random) -> list[str]:
    labels = list(string.ascii_uppercase[:n])
    rng.shuffle(labels)
    return labels


def write_outputs(ctx: list[dict], answers_by_model: dict[str, list[dict]],
                  models: list[str], rng: random.Random, args) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    blind_md = [
        f"# LLM 답변 품질 비교 (블라인드) — {ts}", "",
        f"- 프롬프트: **{args.prompt}**",
        f"- 검색: `{args.embed_model}` / `{args.store}` / top-k={args.k} (모든 모델 동일)",
        f"- 모델 {len(models)}종, 라벨은 **질문마다 랜덤 셔플**",
        "- 임계값 미적용 (유사도 낮아도 LLM 호출)", "",
        "## 채점 기준", "",
        "| 항목 | 점수 |",
        "| --- | --- |",
        "| 정확성 (검색 FAQ·정책과 일치) | 0 / 1 / 2 |",
        "| 할루시네이션 (없는 숫자·조건 지어냄) | 0 / -3 |",
        "| 근거 충실 (검색된 FAQ 근거로 답) | 0 / 1 / 2 |",
        "| 한국어 자연스러움 | 0 / 1 / 2 |",
        "| 거부·되묻기 (모르면 모른다) | 0 / 1 / 2 |", "",
        "---", "",
    ]
    named_md = [f"# LLM 답변 품질 비교 (모델명 공개) — {ts}", "",
                f"- 프롬프트: **{args.prompt}**", "", "---", ""]
    key: dict[str, dict] = {}
    score_rows: list[list[str]] = []

    for i, c in enumerate(ctx, start=1):
        res = c["q"]
        answers = [answers_by_model[m][i - 1] for m in models]
        labels = blind_labels(len(answers), rng)
        lab_map = {lab: a["model"] for lab, a in zip(labels, answers)}
        key[f"Q{i}"] = {"question_id": res["id"], **lab_map}

        header = (f"## Q{i} · {res['id']} [{res['kind']}/{res.get('type','')}] "
                  f"— top_score {c['top_score']}")
        qline = f"**질문**: {res['query']}"
        checkline = f"**확인 포인트**: {res.get('check','')}"
        srcblock = f"<details><summary>검색된 FAQ</summary>\n\n{c['sources_md']}\n\n</details>"

        for md, blind in ((blind_md, True), (named_md, False)):
            md += [header, "", qline, ""]
            if not blind:
                md += [checkline, ""]
            md += [srcblock, ""]
            items = sorted(zip(labels, answers), key=lambda x: x[0]) if blind \
                else [(a["model"], a) for a in answers]
            for lab, a in items:
                tag = f"답변 {lab}" if blind else f"[{a['model']}]"
                flags = []
                if a["error"]:
                    flags.append("오류")
                if a["had_think_block"]:
                    flags.append("think 블록 제거됨")
                flagstr = f"  _({', '.join(flags)})_" if flags else ""
                md += [f"**{tag}**{flagstr}", "", a["text"], ""]
            if blind:
                cells = "  ".join(f"{lab}[정확: /할루: ]" for lab in sorted(labels))
                md += [f"채점 → {cells}", ""]
            md += ["---", ""]

        for lab in sorted(labels):
            score_rows.append([f"Q{i}", res["id"], lab, "", "", "", "", "", ""])

    out = config.RESULTS_DIR
    sfx = f"__{args.prompt}"
    with open(os.path.join(out, f"llm_quality_blind{sfx}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(blind_md) + "\n")
    with open(os.path.join(out, f"llm_quality_named{sfx}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(named_md) + "\n")
    with open(os.path.join(out, f"llm_quality_key{sfx}.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": ts, "prompt": args.prompt, "models": models, "key": key},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, f"llm_quality_score{sfx}.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Q", "question_id", "label", "정확성(0-2)", "할루시네이션(0/-3)",
                    "근거(0-2)", "자연스러움(0-2)", "거부되묻기(0-2)", "메모"])
        w.writerows(score_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 답변 품질 비교 하네스")
    ap.add_argument("--models", nargs="*",
                    default=["exaone3.5:7.8b", "llama3.1:8b", "gemma3:4b"])
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--store", default="pgvector")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default="base", choices=list(prompts.PROMPTS),
                    help="app/prompts.py 의 시스템 프롬프트 변형")
    args = ap.parse_args()
    system_prompt = prompts.PROMPTS[args.prompt]

    questions = load_questions()
    rng = random.Random(args.seed)
    print(f"프롬프트: {args.prompt}")
    print(f"질문 {len(questions)}건 × 모델 {len(args.models)}종 = "
          f"{len(questions) * len(args.models)}개 답변 생성\n")

    ctx = retrieve_all(questions, embed_model=args.embed_model, store=args.store, k=args.k)

    answers_by_model: dict[str, list[dict]] = {}
    for m in args.models:
        print(f"=== {m} ===")
        rows = []
        for i, c in enumerate(ctx, start=1):
            a = asyncio.run(generate(system_prompt, c["user_prompt"], m))
            rows.append(a)
            note = []
            if a["error"]:
                note.append("오류")
            if a["had_think_block"]:
                note.append("think")
            if a["rejected_phrasing"]:
                note.append("거부문구")
            print(f"  [{i:2}/{len(ctx)}] {c['q']['id']:<8} {a.get('total_ms') or '?':>7}ms"
                  f"{('  · ' + ','.join(note)) if note else ''}")
        answers_by_model[m] = rows
        print()

    write_outputs(ctx, answers_by_model, args.models, rng, args)
    sfx = f"__{args.prompt}"
    print("저장:")
    for stem in ("llm_quality_blind", "llm_quality_named",
                 "llm_quality_key", "llm_quality_score"):
        ext = ".json" if stem.endswith("key") else (".csv" if stem.endswith("score") else ".md")
        print(f"  {os.path.join(config.RESULTS_DIR, stem + sfx + ext)}")
    print(f"\n채점: llm_quality_blind{sfx}.md 를 보고 llm_quality_score{sfx}.csv 를 채운다.")


if __name__ == "__main__":
    main()
