.PHONY: setup test test-integration lint fmt clean

## Create virtual env and install all dependencies
setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt
	cp -n .env.example .env || true
	@echo "\n✓ Setup complete. Edit .env with your API keys, then: source .venv/bin/activate"

## Run unit tests with coverage (no real API keys needed)
test:
	pytest tests/ --ignore=tests/integration -v --cov=utils --cov-report=term-missing --cov-fail-under=80

## Run integration tests against real APIs (requires .env with valid keys)
test-integration:
	pytest tests/integration/ -v -s

## Lint with ruff
lint:
	ruff check utils/ tests/

## Auto-format with black
fmt:
	black utils/ tests/

## Remove generated artifacts
clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
