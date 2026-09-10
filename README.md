# LLM RAG 통신사 상담 - 설계 검증용 스파이크

본 프로젝트(`PersenalDocs/Structure.md`)의 **6.1 미결 사항**을 실측으로 좁히기 위한 테스트 프로젝트입니다.

| 확인하려는 것 | 방법 |
| --- | --- |
| 임베딩 모델 (bge-m3 / multilingual-e5-large / nomic-embed-text) | 평가 질문 30건으로 Recall@k, MRR 측정 |
| Vector DB (PostgreSQL+pgvector / PostgreSQL+Qdrant) | 같은 벡터를 넣고 검색 지연과 운영 복잡도 비교 |
| CPU 양자화 LLM 실사용 가능성 | 첫 토큰 도달 시간, 전체 응답 시간 측정 (NFR-1.3, 1.4) |
| RAG 파이프라인 전 구간 | 질문 → 임베딩 → 검색 → 프롬프트 → 스트리밍 답변 |

호스트에는 **Docker 하나만** 있으면 됩니다. Python·Node·Ollama 모두 컨테이너 안에 있습니다.

---

## 빠른 시작

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

```bash
bash scripts/setup.sh      # Git Bash / WSL / macOS
```

스크립트가 하는 일: `.env` 생성 → 컨테이너 기동 → 모델 4종 내려받기(4~5GB) → FAQ 100건 적재 → 비교 실험 실행.

끝나면:
- 챗봇 <http://localhost:8000>
- 비교 결과 `results/comparison.md`
- API 문서 <http://localhost:8000/docs>

## 수동 실행

```bash
cp .env.example .env
docker compose up -d --build

# 모델 (첫 1회만)
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull zylonai/multilingual-e5-large
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull exaone3.5:7.8b

# 적재 - 임베딩은 모델당 1회만 계산해 results/ 에 캐시하고 두 스토어가 공유한다
docker compose exec api python -m app.ingest
docker compose exec api python -m app.ingest --models bge-m3 --stores qdrant   # 일부만

# 비교 실험
docker compose exec api python -m app.evaluate
docker compose exec api python -m app.evaluate --k 5 --models bge-m3

# LLM 응답 시간 측정 (NFR-1.3, 1.4)
docker compose exec api python -m app.llm_bench
docker compose exec api python -m app.llm_bench --model exaone3.5:2.4b
```

## 구조

```
docker-compose.yml         postgres(pgvector) / qdrant / ollama / api
db/init/01_init.sql        faq 테이블, experiment_run 테이블, vector 확장
data/faqs.json             가상 통신사 FAQ 100건 (카테고리 10종 x 10건)
data/eval_questions.json   평가 질문 30건 + 정답 FAQ id
docs/POLICY.md             가상 통신사 정책 기준 문서 (FAQ 의 단일 출처)
docs/EXPERIMENT.md         실험 설계와 판단 기준, 결과 기록
app/
  config.py                임베딩/스토어 레지스트리. 교체는 여기 한 곳
  ollama_client.py         임베딩 + 채팅 스트리밍
  stores/                  base(인터페이스) / pgvector_store / qdrant_store
  rag.py                   검색 → 프롬프트 → 생성
  ingest.py                적재 + 중복 검사
  evaluate.py              Recall@k, MRR, 지연 측정 → results/comparison.md
  main.py                  FastAPI (+ 정적 프론트 서빙)
  smoke_test.py            Ollama 없이 DB 배선만 검증
  llm_bench.py             CPU LLM 첫 토큰/전체 응답 시간 측정
  scale_bench.py           1,000~10,000건 규모별 pgvector vs Qdrant 부하 비교
  static/index.html        챗봇 화면 (Node 불필요)
results/                   임베딩 캐시(.npz), 비교 결과(.md/.json)
```

## API

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| GET | `/api/config` | 선택 가능한 임베딩/스토어, 적재 건수, 기본값 |
| POST | `/api/search` | 검색만 수행 (조합 비교용) |
| POST | `/api/chat` | SSE 스트리밍 답변. `retrieval` → `token`* → `done`\|`error` |
| GET | `/api/results` | `evaluate` 실행 결과 JSON |

```bash
curl -s localhost:8000/api/search -H "Content-Type: application/json" \
  -d '{"question":"약정 중에 해지하면 위약금 얼마예요","embed_model":"bge-m3","store":"qdrant","top_k":3}'
```

## 본 프로젝트와의 대응

이 스파이크는 Python 이지만 계층 구조는 본 프로젝트(Spring Boot)와 1:1로 대응됩니다.

| 스파이크 | 본 프로젝트 |
| --- | --- |
| `app/main.py` | `@RestController` |
| `app/rag.py` | `ChatService` |
| `app/stores/base.py` | `VectorSearchRepository` 인터페이스 |
| `app/stores/pgvector_store.py` / `qdrant_store.py` | 구현체 2종 |
| `app/config.py` | `application.yml` + `@ConfigurationProperties` |

넘어가는 것은 코드가 아니라 **결정**입니다: 어떤 임베딩, 어떤 Vector DB, top-k 얼마, 임계값 얼마.

## 알려진 제약

- `multilingual-e5-large` 는 Ollama 공식 라이브러리에 없어 커뮤니티 태그(`zylonai/multilingual-e5-large`)를 기본값으로 씁니다. 내려받기가 실패하면 `.env` 의 `E5_OLLAMA_MODEL` 을 다른 태그로 바꾸세요.
- CPU 환경에서 임베딩 100건은 모델에 따라 1~10분 걸립니다. `results/embeddings_*.npz` 캐시가 있으면 건너뜁니다.
- 매장 안내(FR-C), 관리자(FR-D), 세션/이력(FR-B8)은 이 스파이크 범위 밖입니다.

## 문제가 생기면

```bash
docker compose ps                      # 컨테이너 상태
docker compose logs -f api             # API 로그
docker compose run --rm --no-deps api python -m app.smoke_test   # 모델 없이 DB 배선만 검증
docker compose down -v                 # 볼륨까지 초기화 후 재시작
```

- **`Name or service not known`**: postgres 컨테이너가 죽어 있습니다. `docker compose logs postgres` 를 확인하세요.
- **임베딩이 너무 느림**: 정상입니다. CPU에서 bge-m3 는 문장 1건당 수백 ms 걸립니다. 적재는 1회만 하면 되고 `results/embeddings_*.npz` 에 캐시됩니다.
- **`model not found`**: 해당 모델을 아직 안 받았습니다. `docker compose exec ollama ollama list` 로 확인하세요.
