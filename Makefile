# ============================================================
# Quant Backtest — Makefile
#
# make test          Run all tests
# make lint          Run ruff linter
# make format        Auto-format code
# make check         lint + typecheck + test
# make backtest S=zzh7.3   Run single backtest
# make report S=zzh7.3     Run single report
# make report-all    Compare all strategies (quick)
# make clean         Remove caches
# ============================================================

S ?= zzh7.3
P ?= 大蓝筹
PY = python3

.PHONY: test lint format typecheck check backtest report report-all clean

# ── Quality ─────────────────────────────────────────────────

install:
	$(PY) -m pip install -e ".[dev,ml]"

test:
	$(PY) -m pytest tests/ -v

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

typecheck:
	$(PY) -m mypy analytics/ backend/ --ignore-missing-imports

check: lint typecheck test
	@echo "All checks passed."

# ── Backtest ───────────────────────────────────────────────

backtest:
	$(PY) run_save.py $(S) --pool $(P)

backtest-all-zzh:
	for s in zzh0.1 zzh1.0 zzh1.9 zzh4.5 zzh7.3 zzhX.0 zzhY.3 zzhY.4; do \
		$(PY) run_save.py $$s --pool $(P); \
	done

# ── Report ─────────────────────────────────────────────────

report:
	$(PY) report.py results/$(S)_$(P).json --benchmark hs300 --scenario

report-full:
	$(PY) report.py results/$(S)_$(P).json --benchmark hs300 --scenario --walk-forward --strategy $(S) --pool $(P)

report-all:
	$(PY) report.py $$(ls results/*_大蓝筹.json 2>/dev/null | head -15) --quick --benchmark hs300 --scenario

# ── Research Agent ──────────────────────────────────────────

research:
	$(PY) -m backend.research_bridge $$(ls results/*_大蓝筹.json 2>/dev/null | head -3) --benchmark hs300 --output /tmp/research_prompt.txt
	@echo "Prompt saved to /tmp/research_prompt.txt"
	@echo "Send to LLM: cat /tmp/research_prompt.txt | your-llm-cli"

# ── Cleanup ────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache
