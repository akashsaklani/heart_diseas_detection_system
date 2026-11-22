# Heart Disease Detection Using Machine Learning

B.Tech Internship Project (Grad Guru, AI/ML with Python) · Akash Saklani · Government College Dharamshala, HPTU

Random Forest classifier that predicts a class (0 or 1, as defined by the dataset's `target` label) from 11 patient health parameters.
Includes the full pipeline from the presentation plus a small web app and REST API.

> **Educational project for preliminary risk assessment only. It is not a medical diagnosis.**

## Important: dataset

The real UCI `heart.csv` could not be bundled. Put it at `data/raw/heart.csv`
(see `data/raw/README.md`). Until then the code falls back to a **synthetic demo file** and prints a warning;
metrics from that file are **not** your project's real results.

## Quick start

```bash
pip install -r requirements.txt
# put real heart.csv in data/raw/   (or: python scripts/make_synthetic_data.py for a demo)
PYTHONPATH=src python -m heart_disease.train --compare   # train, evaluate, plots, 5-model comparison
python -m pytest -q                                        # tests
python app/app.py                                          # http://127.0.0.1:5000
```

CLI prediction:
```bash
PYTHONPATH=src python -m heart_disease.predict --age 45 --sex 1 --cp 0 --trestbps 130 \
  --chol 230 --fbs 0 --restecg 1 --thalach 150 --exang 0 --oldpeak 1.0 --slope 1
```

API:
```bash
curl -X POST localhost:5000/api/predict -H 'Content-Type: application/json' \
  -d '{"age":45,"sex":1,"cp":0,"trestbps":130,"chol":230,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.0,"slope":1}'
```

## Structure

```
heart-disease-detection/
├── data/raw/                 heart.csv (you add) · synthetic demo file
├── src/heart_disease/
│   ├── config.py             paths, features, ranges, seeds
│   ├── data.py               load, clean, split (80:20, seed 42)
│   ├── eda.py                analysis plots
│   ├── model.py              Random Forest + 4 comparison models
│   ├── evaluate.py           metrics, confusion matrix, CV comparison
│   ├── train.py              training CLI
│   └── predict.py            validation + inference
├── app/                      Flask UI + REST API
├── notebooks/                walkthrough notebook (as in the viva)
├── tests/                    pytest suite (12 tests)
├── models/                   saved model (generated)
├── reports/                  metrics.json, model_comparison.csv, figures/
├── docs/architecture.md      diagrams and design notes
├── Dockerfile · Makefile · requirements.txt
```

See `docs/architecture.md` for the workflow diagram and design decisions.

## Pipeline (matches the slides)

1. Load `heart.csv` → 2. check missing, drop duplicates (1025 → 302) → 3. EDA plots →
4. 80:20 split (241 / 61, `random_state=42`) → 5. Random Forest (`n_estimators=100`) →
6. accuracy / precision / recall / F1 → 7. patient input → 8. class 0/1 output.

## Viva tip

Say your actual accuracy from the **real** dataset run (`reports/metrics.json` after training on real `heart.csv`).
Mention that 5-fold CV and the 5-algorithm comparison are your "future scope" extension, now implemented.
