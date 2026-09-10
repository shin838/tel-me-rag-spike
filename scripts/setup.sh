#!/usr/bin/env bash
# 전체 환경을 한 번에 준비한다.  실행:  bash scripts/setup.sh
set -euo pipefail

echo "[1/5] .env 준비"
[ -f .env ] || { cp .env.example .env; echo "  .env 생성됨"; }

echo "[2/5] 컨테이너 기동 (postgres / qdrant / ollama / api)"
docker compose up -d --build

echo "[3/5] 모델 내려받기 - 총 4~5GB, 회선에 따라 10~30분"
for m in bge-m3 zylonai/multilingual-e5-large nomic-embed-text exaone3.5:7.8b; do
  echo "  pull $m ..."
  docker compose exec -T ollama ollama pull "$m" || \
    echo "  [경고] $m 내려받기 실패. docs/EXPERIMENT.md 의 대체 태그를 확인하세요."
done

echo "[4/5] FAQ 100건 적재 (임베딩 3종 x 스토어 2종)"
docker compose exec -T api python -m app.ingest

echo "[5/5] 비교 실험 실행"
docker compose exec -T api python -m app.evaluate

echo
echo "완료. 챗봇: http://localhost:8000   결과: results/comparison.md"
