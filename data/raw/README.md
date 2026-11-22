# Dataset

Put the **real** UCI Heart Disease `heart.csv` (the file used in the presentation) here as:

    data/raw/heart.csv

Required columns: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, target`
(extra columns such as `ca`, `thal` are ignored).

`heart_synthetic_demo.csv` is **fake, generated data** (`scripts/make_synthetic_data.py`) with the same
schema and the same 1025 → 302 duplicate structure. It only exists so the code runs without the real file.
If `heart.csv` is missing, training falls back to it and prints a warning. **Never report its metrics as real results.**
