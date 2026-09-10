"""Ollama HTTP 클라이언트 (임베딩 + 채팅 스트리밍)."""
from __future__ import annotations

import json
import time
from typing import AsyncIterator, Iterable

import httpx

from . import config


class OllamaError(RuntimeError):
    pass


# ---------------------------------------------------------------- 임베딩(동기)

def embed(texts: Iterable[str], model_name: str, *, is_query: bool) -> tuple[list[list[float]], float]:
    """텍스트 목록을 임베딩하고 (벡터목록, 소요 ms)를 돌려준다.

    is_query=True 면 질의용 접두어, False 면 문서용 접두어를 붙인다.
    """
    conf = config.embed_conf(model_name)
    prefix = conf["query_prefix"] if is_query else conf["doc_prefix"]
    payload_input = [prefix + t for t in texts]

    started = time.perf_counter()
    with httpx.Client(timeout=config.EMBED_TIMEOUT_S) as client:
        try:
            r = client.post(
                f"{config.OLLAMA_URL}/api/embed",
                json={"model": conf["ollama"], "input": payload_input},
            )
            r.raise_for_status()
            vectors = r.json()["embeddings"]
        except httpx.HTTPStatusError as e:
            # 구버전 Ollama 는 /api/embeddings(단건, prompt) 만 지원한다.
            if e.response.status_code != 404:
                raise OllamaError(f"임베딩 실패({model_name}): {e.response.text[:300]}") from e
            vectors = []
            for text in payload_input:
                rr = client.post(
                    f"{config.OLLAMA_URL}/api/embeddings",
                    json={"model": conf["ollama"], "prompt": text},
                )
                rr.raise_for_status()
                vectors.append(rr.json()["embedding"])
    elapsed_ms = (time.perf_counter() - started) * 1000
    return vectors, elapsed_ms


def embed_one(text: str, model_name: str, *, is_query: bool) -> tuple[list[float], float]:
    vectors, ms = embed([text], model_name, is_query=is_query)
    return vectors[0], ms


def probe_dim(model_name: str) -> int:
    """모델의 임베딩 차원을 실측한다. 차원을 하드코딩하지 않기 위한 장치."""
    vec, _ = embed_one("차원 확인용 문장", model_name, is_query=False)
    return len(vec)


# ---------------------------------------------------------------- 채팅(비동기 스트리밍)

async def chat_stream(system: str, user: str, *, model: str | None = None) -> AsyncIterator[dict]:
    """Ollama 채팅을 스트리밍한다. {"type": "token"|"done"|"error", ...} 를 순서대로 내보낸다."""
    model = model or config.LLM_MODEL
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    started = time.perf_counter()
    first_token_ms: float | None = None

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(config.LLM_TIMEOUT_S, connect=10.0)) as client:
            async with client.stream("POST", f"{config.OLLAMA_URL}/api/chat", json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:300]
                    yield {"type": "error", "message": f"LLM 호출 실패({resp.status_code}): {detail}"}
                    return
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000
                        yield {"type": "token", "text": piece}
                    if chunk.get("done"):
                        yield {
                            "type": "done",
                            "first_token_ms": round(first_token_ms or 0, 1),
                            "total_ms": round((time.perf_counter() - started) * 1000, 1),
                            "model": model,
                        }
                        return
    except httpx.TimeoutException:
        # NFR-2.5: '느림'과 '실패'를 구분해 전달한다.
        yield {"type": "error", "message": f"응답이 {config.LLM_TIMEOUT_S:.0f}초를 넘겨 중단했습니다. 다시 시도해 주세요.", "reason": "timeout"}
    except httpx.HTTPError as e:
        yield {"type": "error", "message": f"LLM 서버에 연결하지 못했습니다: {e}", "reason": "connection"}


def list_models() -> list[str]:
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{config.OLLAMA_URL}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
    except httpx.HTTPError:
        return []
