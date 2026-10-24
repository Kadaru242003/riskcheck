PY ?= python

.PHONY: install data portfolios backtest drift copilot report test lint all

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m riskcheck.data

portfolios:
	@echo "Part 1: not implemented yet"

backtest:
	@echo "Parts 2-3: not implemented yet"

drift:
	@echo "Part 4: not implemented yet"

copilot:
	@echo "Part 5: not implemented yet"

report:
	@echo "Report generation: not implemented yet"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

all: data portfolios backtest drift copilot report
