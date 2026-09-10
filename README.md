# Tel-Me RAG 검증 — 임베딩 / Vector DB / 유사도 임계값 / LLM

Tel-Me(가상 통신사 AI 상담) 설계의 미결 항목을 실측으로 좁히기 위한 테스트 저장소.
RAG 파이프라인(질문 → 임베딩 → 벡터 검색 → LLM 답변)을 Docker로 재현하고 각 부품을 비교·측정한다.

| 검증 항목 | 방법 | 결과 파일 |
| --- | --- | --- |
| 임베딩 모델 (bge-m3 / e5-large / nomic) | 평가 질문 30건으로 Recall@k, MRR | `results/comparison.md` |
| Vector DB (pgvector / Qdrant) | 같은 벡터 넣고 검색 지연·운영 비교 | `results/comparison.md` |
| **유사도 임계값 (FR-B7)** | 답 없는 질문 15건으로 `SIM_THRESHOLD` 스윕 | `results/threshold_sweep.md` |
| **LLM 품질** | 질문 13건 × 모델 N종 블라인드 채점, 프롬프트 3버전 A/B | `results/llm_quality_*__{base,v2,v3}.md` |
| LLM 속도 | 첫 토큰 / 전체 응답 시간 (NFR-1.3, 1.4) | `results/llm_bench_*.json` |

호스트에는 Docker만 있으면 된다. Python·Ollama·DB 모두 컨테이너 안에 있다.

---

## 결론 요약 (잠정)

| 항목 | 결론 | 근거 |
| --- | --- | --- |
| 임베딩 모델 | **bge-m3** | Recall@3 90.0%, e5 대비 임베딩 3.4배 빠름. nomic은 Recall@3 20% + 점수 변별력 없음으로 탈락 |
| Vector DB | **PostgreSQL + pgvector** | 검색 품질 동일, FR-D3 동기화를 트랜잭션으로 해결. 규모(1,000건)에 Qdrant 이점 없음 |
| top-k | 3 | Recall@3 90% |
| 유사도 임계값 | **0.48~0.50** (잠정) | 답 있는/없는 질문 점수 분포가 0.50~0.70에서 크게 겹침 → 벡터 단독으로 "통신스러운 범위 밖 질문"은 못 거름. 정밀 거부는 하이브리드 검색 / 의도 분류기(P2) 필요 |
| LLM | **exaone3.5:7.8b** 우세 | 프롬프트 3버전 모두 지시 준수·형식 일관. 소형 모델(gemma 4B·llama 8B)은 "거부 vs 답변" 경계를 못 잡음. qwen3는 thinking 모드로 속도 탈락 |
| 응답 방식 | SSE 스트리밍 | 전체 응답 5~12초 |

- **재현 검증**: 동일 데이터·모델이면 Recall·MRR은 결정적이라, 환경(macOS)이 달라도 수치가 일치함을 확인 (nomic 중복 후보 92쌍까지 동일) → 설정 정상.
- **하드웨어**: macOS / M5 Pro는 Docker가 Apple GPU를 못 써서 CPU만 사용 → 임베딩 느림(bge-m3 질의 임베딩 46ms→170ms), LLM은 빠름(통합 메모리 대역폭). **배포 대상 EC2에서 재측정 필요.**
- **미결**: 실제 FAQ 1,000건 확장 후 재측정 / retrieval 실패 케이스(Q-08, Q-30) 점검 / LLM 최종 확정은 재검증(질문 확대·블라인드 다인 채점) 후.

---

## 빠른 시작

```bash
cp .env.example .env
docker compose up -d --build

# 임베딩 모델 (3종)
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull zylonai/multilingual-e5-large
docker compose exec ollama ollama pull nomic-embed-text

# LLM (테스트한 4종)
docker compose exec ollama ollama pull exaone3.5:7.8b
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull gemma3:4b

docker compose exec api python -m app.ingest    # FAQ 100건 적재 (CPU라 모델당 1~10분)
```

챗봇 <http://localhost:8000> · API 문서 <http://localhost:8000/docs>

> 내 하드웨어에서 임베딩 속도를 재려면 `results/embeddings_*.npz` 캐시를 지운 뒤 `app.ingest` 를 실행한다. 캐시가 있으면 임베딩을 건너뛴다.

---

## 테스트 실행

### 1. 임베딩 모델 / Vector DB

```bash
docker compose exec api python -m app.evaluate           # 6개 조합 Recall@k, MRR, 지연
docker compose exec api python -m app.scale_bench        # 1,000~10,000건 규모별 (합성 벡터)
```
→ `results/comparison.md` (종합 표 + 검색 실패 사례)

### 2. 유사도 임계값 (FR-B7)

```bash
docker compose exec api python -m app.threshold_eval     # 답있음 30 + 답없음 15의 top-1 유사도 → 임계값 스윕
```
→ `results/threshold_sweep.md` (분포 요약 + 임계값별 통과율/거부율 + 제안값)

- 답 없는 질문 세트: `data/eval_out_of_scope.json` (무관 5 / 범위밖 5 / 없는정책 5)
- 옵션: `--model`, `--store`, `--lo/--hi/--step`, `--min-pass`

### 3. LLM 속도

```bash
docker compose exec api python -m app.llm_bench --model exaone3.5:7.8b
docker compose exec api python -m app.llm_bench --model gemma3:4b
```
→ `results/llm_bench_<model>.json` (콜드/워밍 구분, 첫 토큰·전체 시간)

### 4. LLM 품질 비교 하네스

```bash
docker compose exec api python -m app.llm_quality --prompt base   # 기본 프롬프트
docker compose exec api python -m app.llm_quality --prompt v2     # 규칙 6개로 보강
docker compose exec api python -m app.llm_quality --prompt v3     # v2에서 규칙 2만 단순화
```

질문 13건(답있음 10 + 답없음 3) × 모델 3종. **검색 결과는 고정하고 LLM만 교체** → LLM만의 차이를 본다. 임계값 미적용(유사도 낮아도 호출해서 "지어내는지 vs 거부하는지" 확인).

산출물 (프롬프트 변형별로 파일명 분리):

| 파일 | 용도 |
| --- | --- |
| `llm_quality_blind__<변형>.md` | 채점용 — 모델명 A/B/C 로 가림, 질문마다 셔플 |
| `llm_quality_named__<변형>.md` | 모델명 공개 (채점 후 확인) |
| `llm_quality_key__<변형>.json` | 정답 키 |
| `llm_quality_score__<변형>.csv` | 채점 입력 템플릿 |

- 프롬프트 변형은 `app/prompts.py` 의 `base` / `v2` / `v3`. 새 변형은 `PROMPTS` dict 에 등록.
- **실험 요약**: 형식 문제(인사말·참고자료 노출)는 프롬프트로 해결됨. "모르면 거부 vs 아는 건 답변" 경계는 소형 모델(gemma 4B)이 프롬프트로 못 잡음 — 규칙을 조이면 "확인이 어렵습니다" 남발, 풀면 할루시네이션. exaone 만 세 버전 일관되게 지시 준수.

---

## 구조

```
docker-compose.yml          postgres(pgvector) / qdrant / ollama / api
db/init/01_init.sql         faq 테이블, experiment_run 테이블, vector 확장
data/
  faqs.json                 가상 통신사 FAQ 100건 (카테고리 10종 × 10)
  eval_questions.json       평가 질문 30건 + 정답 FAQ id (Recall 용)
  eval_out_of_scope.json    답 없는 질문 15건 (임계값 용)
  eval_quality.json         LLM 품질 채점 질문 13건 + check 필드
docs/
  POLICY.md                 가상 통신사 정책 문서 (FAQ 의 단일 출처)
  EXPERIMENT.md             임베딩/DB 실험 설계·판단 기준
app/
  config.py                 임베딩/스토어 레지스트리
  ollama_client.py          임베딩 + 채팅 스트리밍
  rag.py                    검색 → 프롬프트 → 생성 (SYSTEM_PROMPT 원본)
  prompts.py                시스템 프롬프트 변형 base / v2 / v3
  stores/                   pgvector_store / qdrant_store
  ingest.py                 적재 + 중복 검사
  evaluate.py               Recall@k, MRR, 지연 → comparison.md
  scale_bench.py            규모별 부하 (합성 벡터)
  llm_bench.py              LLM 응답 시간
  threshold_eval.py         유사도 임계값 스윕
  llm_quality.py            LLM 품질 비교 하네스
  main.py                   FastAPI + 정적 프론트
results/                    결과 (.md / .json). 임베딩 캐시(.npz)는 git 제외
```

---

## API

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| GET | `/api/config` | 선택 가능한 임베딩/스토어, 적재 건수 |
| POST | `/api/search` | 검색만 수행 |
| POST | `/api/chat` | SSE 스트리밍 답변 (`retrieval` → `token`* → `done`\|`error`) |
| GET | `/api/results` | evaluate 결과 JSON |

## 본 프로젝트(Spring Boot)와의 대응

| 스파이크 | 본 프로젝트 |
| --- | --- |
| `app/main.py` | `@RestController` |
| `app/rag.py` | `ChatService` |
| `app/stores/base.py` | `VectorSearchRepository` 인터페이스 |
| `app/config.py` / `app/prompts.py` | `application.yml` + `@ConfigurationProperties` |

넘어가는 것은 코드가 아니라 **결정**: 어떤 임베딩, 어떤 Vector DB, top-k, 임계값, 어떤 LLM.

## 알려진 제약

- `multilingual-e5-large` 는 Ollama 공식에 없어 커뮤니티 태그(`zylonai/multilingual-e5-large`)를 쓴다. 실패 시 `.env` 의 `E5_OLLAMA_MODEL` 변경.
- CPU 환경 임베딩 100건은 모델당 1~10분. `results/embeddings_*.npz` 캐시가 있으면 건너뛴다.
- `qwen3:8b` 는 thinking 모드가 기본이라 첫 토큰이 매우 느리다. `think: false` 없이는 속도 비교에서 제외.
- 검색 정확도·품질 테스트는 FAQ 100건 기준. 1,000건 확장 시 재측정 필요.
- 매장 안내(FR-C), 관리자(FR-D), 세션/이력(FR-B8)은 범위 밖.

## 문제 해결

```bash
docker compose ps
docker compose logs -f api
docker compose run --rm --no-deps api python -m app.smoke_test   # 모델 없이 DB 배선만 검증
docker compose down -v                                           # 볼륨까지 초기화
```

- **포트 5432 충돌**: 다른 postgres 컨테이너 중지(`docker stop <name>`) 또는 `.env` 의 `POSTGRES_PORT` 변경
- **`model not found`**: `docker compose exec ollama ollama list` 로 확인
- **macOS에서 느림**: Docker Desktop이 Apple GPU를 못 써서 Ollama가 CPU만 사용. 정상이며, 배포(EC2 CPU)와는 오히려 비슷한 조건
