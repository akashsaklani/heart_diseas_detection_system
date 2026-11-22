.PHONY: install demo-data train compare test serve docker

install:
	pip install -r requirements.txt

demo-data:   ## only if you do NOT have the real heart.csv
	python scripts/make_synthetic_data.py

train:
	PYTHONPATH=src python -m heart_disease.train

compare:
	PYTHONPATH=src python -m heart_disease.train --compare

test:
	python -m pytest -q

serve:
	python app/app.py

docker:
	docker build -t heart-disease-detection . && docker run -p 5000:5000 heart-disease-detection
