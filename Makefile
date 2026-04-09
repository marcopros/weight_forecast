.PHONY: clean features train predict serve test monitor dashboard all

# Full pipeline
all: clean features train predict

clean:
	python -m src.data.clean

features:
	python -m src.data.features

train:
	python -m src.models.train

predict:
	python -m src.models.predict

# Serving
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Monitoring
monitor:
	python -m src.monitoring.drift
	python -m src.monitoring.performance

dashboard:
	streamlit run src/monitoring/dashboard.py

# Testing
test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

# DVC pipeline (alternative to make all)
dvc:
	dvc repro

# Docker
docker-build:
	docker build -f docker/Dockerfile.api -t forecast-api .

docker-run:
	docker run -p 8000:8000 forecast-api

docker-compose:
	cd docker && docker-compose up --build
