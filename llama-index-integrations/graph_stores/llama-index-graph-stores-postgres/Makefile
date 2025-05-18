.PHONY: format lint test

format:
	black .
	isort .

lint:
	mypy .
	black . --check
	isort . --check

test:
	pytest -xvs tests/