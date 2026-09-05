.PHONY: install verify benchmark serve test icons

install:
	pip install -e ".[dev]"

verify: benchmark test
	@echo "verify OK"
	@python3 -c "import json; r=json.load(open('evidence/benchmark-report.json')); print('detection', r.get('aggregate_detection')); print('id', r.get('aggregate_identification'))"

benchmark:
	python3 -m shipwreck_sat_finder benchmark --out evidence/benchmark-report.json

test:
	python3 -m pytest -q

serve:
	python3 -m shipwreck_sat_finder serve --host 0.0.0.0 --port 7860
