.PHONY: format lint test

format:
	black .
	isort .

lint:
	mypy .
	black . --check
	isort . --check
	flake8 .

test:
	pytest -xvs tests/