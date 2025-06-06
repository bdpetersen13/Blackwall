.PHONY: help install install-dev test lint format clean build run

help:
	@echo "Available commands:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linters"
	@echo "  make format       Format code"
	@echo "  make clean        Clean build artifacts"
	@echo "  make build        Build distribution packages"
	@echo "  make run          Run the CLI tool"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=blackwall --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/

build: clean
	python setup.py sdist bdist_wheel

run:
	python -m blackwall

download-models:
	python scripts/download_models.py