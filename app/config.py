"""스파이크 전역 설정.

임베딩 모델과 Vector DB를 '설정으로 교체 가능'하게 유지한다(NFR-5.1과 동일한 원칙).
"""
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

PG = dict(
    host=os.getenv("POSTGRES_HOST", "postgres"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    user=os.getenv("POSTGRES_USER", "raguser"),
    password=os.getenv("POSTGRES_PASSWORD", "ragpass"),
    dbname=os.getenv("POSTGRES_DB", "ragdb"),
)

LLM_MODEL = os.getenv("LLM_MODEL", "exaone3.5:2.4b")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "60"))   # NFR-2.1 (스파이크는 CPU라 넉넉히)
EMBED_TIMEOUT_S = float(os.getenv("EMBED_TIMEOUT_S", "120"))

DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "bge-m3")
DEFAULT_STORE = os.getenv("DEFAULT_STORE", "pgvector")

TOP_K = int(os.getenv("TOP_K", "3"))                       # NFR-1.6
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.45"))  # FR-B7 (코사인 유사도)
MAX_QUESTION_LEN = int(os.getenv("MAX_QUESTION_LEN", "500"))  # NFR-2.4

DATA_DIR = os.getenv("DATA_DIR", "/srv/data")
RESULTS_DIR = os.getenv("RESULTS_DIR", "/srv/results")

# 임베딩 모델 레지스트리.
# e5 계열은 query/passage 접두어가 성능에 크게 영향을 준다. 비교 공정성을 위해 모델별로 명시한다.
EMBED_MODELS = {
    "bge-m3": {
        "key": "bge_m3",
        "ollama": os.getenv("BGE_OLLAMA_MODEL", "bge-m3"),
        "query_prefix": "",
        "doc_prefix": "",
        "note": "다국어 검색 특화. 접두어 불필요.",
    },
    "multilingual-e5-large": {
        "key": "e5_large",
        # ollama 공식 라이브러리에 없어 커뮤니티 태그를 기본값으로 둔다. 실패 시 .env 로 교체.
        "ollama": os.getenv("E5_OLLAMA_MODEL", "zylonai/multilingual-e5-large"),
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "note": "query:/passage: 접두어 필수. 빠뜨리면 성능이 크게 떨어진다.",
    },
    "nomic-embed-text": {
        "key": "nomic",
        "ollama": os.getenv("NOMIC_OLLAMA_MODEL", "nomic-embed-text"),
        "query_prefix": "search_query: ",
        "doc_prefix": "search_document: ",
        "note": "경량(768차원). 영어 중심이라 한국어 성능 확인이 목적.",
    },
}

STORES = ["pgvector", "qdrant"]


def embed_conf(name: str) -> dict:
    if name not in EMBED_MODELS:
        raise ValueError(f"알 수 없는 임베딩 모델: {name} (가능: {list(EMBED_MODELS)})")
    return EMBED_MODELS[name]
