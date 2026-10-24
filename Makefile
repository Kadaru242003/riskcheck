PY ?= python
COPILOT_MODELS ?= openai/gpt-oss-120b openai/gpt-oss-20b qwen/qwen3.8-27b

.PHONY: install data portfolios forecasts backtest drift copilot-questions copilot-models \
	copilot-pilot copilot copilot-judge copilot-label copilot-report mcp report test lint all

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m riskcheck.data

portfolios:
	$(PY) -m riskcheck.portfolios

forecasts:
	$(PY) -m riskcheck.forecasts

backtest:
	$(PY) -m riskcheck.backtest

drift:
	$(PY) -m riskcheck.drift

copilot-questions:
	$(PY) -m riskcheck.copilot.questions

copilot-models:
	$(PY) -m riskcheck.copilot.runner models

copilot-pilot:
	$(PY) -m riskcheck.copilot.runner pilot --models $(COPILOT_MODELS)

copilot:
	$(PY) -m riskcheck.copilot.runner run --models $(COPILOT_MODELS)

copilot-judge:
	$(PY) -m riskcheck.copilot.analysis judge

copilot-label:
	$(PY) -m riskcheck.copilot.analysis label

copilot-report:
	$(PY) -m riskcheck.copilot.analysis report

mcp:
	$(PY) -m riskcheck.mcp_server

report:
	@echo "Report generation: not implemented yet"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

all: data portfolios forecasts backtest drift copilot report
