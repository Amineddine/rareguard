install:
	pip install -r requirements.txt
	pip install -e .

demo-data:
	python scripts/generate_synthetic_logs.py --out data/synth.jsonl --days 2 --hosts 5 --attack-rate 0.002

train:
	python -m rare_guard.cli train --log data/synth.jsonl --out artifacts --seq-len 24 --epochs 3

calibrate:
	python -m rare_guard.cli calibrate --artifacts artifacts --alpha 0.01

detect:
	python -m rare_guard.cli detect --log data/synth.jsonl --artifacts artifacts --show 20

api:
	uvicorn rare_guard.api:app --reload --port 8000
