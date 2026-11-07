.PHONY: test demo cli
test:
	PYTHONPATH=src python3 -m pytest tests/ -q
demo:
	PYTHONPATH=src python3 scripts/demo.py
cli:
	PYTHONPATH=src python3 -m mdvp.cli --help
