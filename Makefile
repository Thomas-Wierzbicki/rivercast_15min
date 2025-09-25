.PHONY: setup etl qc features train forecast all

setup:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip -r requirements.txt

etl:
	python -m src.etl.load

qc:
	python -m src.etl.qc

features:
	python -m src.features.build_features

train:
	python -m src.models.train_baseline

forecast:
	python -m src.serve.publish

all: etl qc features train forecast
