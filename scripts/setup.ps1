# 전체 환경을 한 번에 준비한다.  실행:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "[1/5] .env 준비" -ForegroundColor Cyan
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "  .env 생성됨" }

Write-Host "[2/5] 컨테이너 기동 (postgres / qdrant / ollama / api)" -ForegroundColor Cyan
docker compose up -d --build

Write-Host "[3/5] 모델 내려받기 - 총 4~5GB, 회선에 따라 10~30분" -ForegroundColor Cyan
$models = @(
  "bge-m3",                          # 임베딩 1 (1024차원)
  "zylonai/multilingual-e5-large",   # 임베딩 2 (1024차원, 커뮤니티 태그)
  "nomic-embed-text",                # 임베딩 3 (768차원)
  "exaone3.5:7.8b"                   # 양자화 LLM (저사양이면 exaone3.5:2.4b)
)
foreach ($m in $models) {
  Write-Host "  pull $m ..." -ForegroundColor DarkGray
  docker compose exec -T ollama ollama pull $m
  if ($LASTEXITCODE -ne 0) { Write-Host "  [경고] $m 내려받기 실패. docs/EXPERIMENT.md 의 대체 태그를 확인하세요." -ForegroundColor Yellow }
}

Write-Host "[4/5] FAQ 100건 적재 (임베딩 3종 x 스토어 2종)" -ForegroundColor Cyan
docker compose exec -T api python -m app.ingest

Write-Host "[5/5] 비교 실험 실행" -ForegroundColor Cyan
docker compose exec -T api python -m app.evaluate

Write-Host ""
Write-Host "완료. 챗봇: http://localhost:8000   결과: results\comparison.md" -ForegroundColor Green
