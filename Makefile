.PHONY: help setup lint test train run-api clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtual environment and install dependencies
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	@echo "\n✅ Setup complete. Activate with: source .venv/bin/activate"

lint: ## Run ruff linter and formatter
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix: ## Auto-fix lint issues
	ruff check --fix src/ tests/
	ruff format src/ tests/

test: ## Run all tests
	pytest tests/

train: ## Train the model
	python -m src.training.train

run-api: ## Start the FastAPI inference server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info .coverage htmlcov/
