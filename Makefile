.PHONY: help setup lint lint-fix test test-cov test-count preprocess data-clean train run-api run-streamlit clean

# Detect OS
ifeq ($(OS),Windows_NT)
    IS_WINDOWS := 1
else
    IS_WINDOWS := 0
endif

help: ## Show this help message
ifeq ($(OS),Windows_NT)
	@powershell -Command "Write-Host ''"
	@powershell -Command "Write-Host 'Available targets:' -ForegroundColor Cyan"
	@powershell -Command "Get-Content Makefile | Select-String '^[a-zA-Z_-]+:.*?## ' | ForEach-Object { $$line = $$_.Line; $$parts = $$line -split ':.*?## '; '{0,-20} {1}' -f $$parts[0], $$parts[1] }"
else
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
endif

setup: ## Create virtual environment and install dependencies
	uv sync --all-extras
	@uv run python -c "import shutil, pathlib; p=pathlib.Path('.env'); p.exists() or (shutil.copy('.env.example', '.env'), print('[OK] .env criado a partir do .env.example -- edite o JWT_SECRET antes de rodar.'))"
	@uv run python -c "print('Setup complete.')"

lint: ## Run ruff linter and formatter
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

lint-fix: ## Auto-fix lint issues
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

data-clean: ## Limpa o .xlsx bruto e gera data/processed/telco_churn_cleaned.csv
	uv run python -m src.data.preprocess

train: preprocess ## Train the model (runs preprocess automatically)
	uv run python -m src.training.train

test: ## Run all tests
	uv run pytest tests/

test-cov: ## Run tests with coverage report
	uv run pytest tests/ --cov=src --cov-report=term-missing

test-count: ## Count total number of tests
ifeq ($(OS),Windows_NT)
	@powershell -Command "uv run pytest tests/ --collect-only -q | Select-Object -Last 1"
else
	uv run pytest tests/ --collect-only -q | tail -1
endif

preprocess: ## Process raw data and save to data/processed/
	uv run python -m src.data.preprocess

run-api: ## Start the FastAPI inference server
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-streamlit: ## Start the Streamlit frontend (requires API running)
	uv run streamlit run src/app/app.py --server.port 8501

clean: ## Remove build artifacts and caches
ifeq ($(OS),Windows_NT)
	@powershell -Command "Get-ChildItem -Recurse -Directory __pycache__ -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Recurse -Directory .ipynb_checkpoints -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Remove-Item dist,build,.coverage,htmlcov -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Filter '*.egg-info' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
else
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info .coverage htmlcov/
endif