.PHONY: install dev model run test smoke compose-build compose-up compose-down clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

model:
	$(PYTHON) scripts/make_model.py

run:
	uvicorn main:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/smoke.py http://127.0.0.1:8000

compose-build:
	docker compose build

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
