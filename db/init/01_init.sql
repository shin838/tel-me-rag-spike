-- pgvector 확장 및 FAQ 원본 테이블
-- 임베딩 테이블은 모델별 차원이 달라 ingest 시점에 동적으로 생성한다(app/stores/pgvector_store.py).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS faq (
    id          TEXT PRIMARY KEY,
    category    TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    keywords    TEXT[] NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_faq_category ON faq (category);
CREATE INDEX IF NOT EXISTS idx_faq_question_trgm ON faq USING gin (question gin_trgm_ops);

-- 실험 결과 기록용(임베딩 모델 x 벡터 DB 조합별 지표)
CREATE TABLE IF NOT EXISTS experiment_run (
    id            BIGSERIAL PRIMARY KEY,
    ran_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    embed_model   TEXT NOT NULL,
    store         TEXT NOT NULL,
    dim           INT  NOT NULL,
    recall_at_1   NUMERIC(5,4),
    recall_at_3   NUMERIC(5,4),
    recall_at_5   NUMERIC(5,4),
    mrr_at_5      NUMERIC(5,4),
    search_ms_avg NUMERIC(10,2),
    search_ms_p95 NUMERIC(10,2),
    index_ms      NUMERIC(10,2),
    embed_ms_avg  NUMERIC(10,2),
    notes         TEXT
);
