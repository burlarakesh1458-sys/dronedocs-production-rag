PY ?= python
export PYTHONPATH := src

.PHONY: install install-dev ingest ask serve ui test lint eval eval-smoke verify-golden ablation charts docker clean

install:            ## runtime deps (CPU torch)
	$(PY) -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps

install-dev: install
	$(PY) -m pip install -r requirements-dev.txt

ingest:             ## build Chroma + BM25 index from data/raw
	$(PY) -m dronedocs.cli -v ingest

ask:                ## make ask Q="What is a nano drone?"
	$(PY) -m dronedocs.cli ask "$(Q)"

serve:              ## FastAPI on :8000 (HTML UI at /)
	$(PY) -m uvicorn dronedocs.api:app --host 0.0.0.0 --port 8000

ui:                 ## Streamlit UI on :8501
	streamlit run ui/streamlit_app.py

test:               ## offline unit tests (no network, no model downloads)
	$(PY) -m pytest

lint:
	ruff check src tests scripts eval ui && ruff format --check src tests scripts eval ui

verify-golden:      ## check every golden evidence quote exists on its cited page/section
	$(PY) scripts/verify_golden.py

eval:               ## full golden-set eval, exits 1 below thresholds (CI gate)
	$(PY) -m dronedocs.cli eval

eval-smoke:
	$(PY) -m dronedocs.cli eval --limit 10 --no-ragas --no-gate

ablation:           ## retrieval ablation (no LLM calls): vector vs bm25 vs hybrid vs hybrid+rerank
	$(PY) scripts/retrieval_ablation.py

charts:
	$(PY) scripts/make_charts.py

docker:
	docker compose up --build

clean:
	rm -rf data/index .cache .pytest_cache .ruff_cache
