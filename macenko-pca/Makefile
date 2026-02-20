.DEFAULT_GOAL := help

.PHONY: help develop test cov lint format fix types docs serve-docs build clean install pre-commit

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode with maturin
	pip install maturin hatch
	maturin develop --release

develop: ## Build the Rust extension in-place for development
	maturin develop --release

test: ## Run tests (builds Rust extension first)
	maturin develop --release
	hatch run test:test

cov: ## Run tests with coverage report
	maturin develop --release
	hatch run test:cov

lint: ## Run Ruff linter
	hatch run lint:check

format: ## Format code with Ruff
	hatch run lint:format

fix: ## Auto-fix lint issues and format
	hatch run lint:all

types: ## Run mypy type checking
	hatch run types:check

docs: ## Build documentation
	hatch run docs:build-docs

serve-docs: ## Serve documentation locally
	hatch run docs:serve-docs

build: ## Build wheel and sdist via maturin
	maturin build --release

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ site/ htmlcov/ target/
	rm -f coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.so' -delete 2>/dev/null || true
	find . -name '*.dylib' -delete 2>/dev/null || true
	find . -name '*.pyd' -delete 2>/dev/null || true

pre-commit: ## Install and run pre-commit hooks
	pre-commit install
	pre-commit run --all-files

cargo-check: ## Run cargo check on the Rust code
	cargo check

cargo-clippy: ## Run cargo clippy lints on the Rust code
	cargo clippy -- -D warnings

cargo-test: ## Run Rust-only unit tests
	cargo test

docker-dev: ## Build and run dev Docker image
	docker build --target dev -t macenko-pca:dev .

docker-prod: ## Build production Docker image
	docker build --target prod -t macenko-pca:prod .
