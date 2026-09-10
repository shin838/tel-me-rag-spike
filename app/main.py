"""FastAPI 진입점. 챗봇 프론트도 여기서 서빙한다(Node 불필요)."""
from __future__ import annotations

import json
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, ollama_client, rag
from .stores import get_store

app = FastAPI(title="한빛모바일 RAG 상담 스파이크", version="0.1.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str = Field(..., description="사용자 질문")
    embed_model: str | None = None
    store: str | None = None
    top_k: int | None = None
    llm_model: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict:
    """프론트가 고를 수 있는 조합과 현재 적재 상태를 함께 준다."""
    loaded = {}
    for model_name, conf in config.EMBED_MODELS.items():
        loaded[model_name] = {
            store: get_store(store).count(conf["key"]) for store in config.STORES
        }
    return {
        "embed_models": [
            {"name": n, "ollama_tag": c["ollama"], "note": c["note"]}
            for n, c in config.EMBED_MODELS.items()
        ],
        "stores": config.STORES,
        "defaults": {
            "embed_model": config.DEFAULT_EMBED_MODEL,
            "store": config.DEFAULT_STORE,
            "top_k": config.TOP_K,
            "llm_model": config.LLM_MODEL,
        },
        "loaded_counts": loaded,
        "ollama_models": ollama_client.list_models(),
        "sim_threshold": config.SIM_THRESHOLD,
    }


@app.post("/api/search")
def search(req: ChatRequest) -> JSONResponse:
    """검색만 수행한다. 임베딩/스토어 조합을 눈으로 비교할 때 쓴다."""
    r = rag.retrieve(
        req.question,
        embed_model=req.embed_model or config.DEFAULT_EMBED_MODEL,
        store_name=req.store or config.DEFAULT_STORE,
        k=req.top_k,
    )
    return JSONResponse(r.to_dict())


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """SSE 스트리밍. 이벤트: retrieval -> token* -> done | error"""

    async def event_source():
        async for event in rag.answer_stream(
            req.question,
            embed_model=req.embed_model or config.DEFAULT_EMBED_MODEL,
            store_name=req.store or config.DEFAULT_STORE,
            k=req.top_k,
            llm_model=req.llm_model,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/results")
def results() -> JSONResponse:
    """python -m app.evaluate 실행 결과를 프론트에서 보여주기 위한 엔드포인트."""
    path = os.path.join(config.RESULTS_DIR, "comparison.json")
    if not os.path.exists(path):
        return JSONResponse({"available": False,
                             "message": "아직 실험 결과가 없습니다. docker compose exec api python -m app.evaluate 를 실행하세요."})
    with open(path, encoding="utf-8") as f:
        return JSONResponse({"available": True, **json.load(f)})
