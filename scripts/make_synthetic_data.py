"""Generate a SYNTHETIC demo dataset with the same schema as the UCI heart.csv.

!! This is NOT real patient data. It exists only so the pipeline, tests and web
!! app can run end-to-end when the real file is not available. For the real
!! results, put the genuine UCI `heart.csv` at data/raw/heart.csv.

Shape mimics the presentation: 1025 rows = 302 unique + 723 duplicates,
12 columns (11 features + target), no missing values.
"""
from pathlib import Path

import numpy as np
import pandas as pd

N_UNIQUE, N_DUP, SEED = 302, 723, 42
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "heart_synthetic_demo.csv"


def make() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = N_UNIQUE
    age = rng.integers(29, 78, n)
    sex = rng.binomial(1, 0.68, n)
    cp = rng.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])
    trestbps = np.clip(rng.normal(131, 17, n), 94, 200).round().astype(int)
    chol = np.clip(rng.normal(246, 52, n), 126, 564).round().astype(int)
    fbs = rng.binomial(1, 0.15, n)
    restecg = rng.choice([0, 1, 2], n, p=[0.48, 0.50, 0.02])
    thalach = np.clip(205 - 0.8 * age + rng.normal(0, 17, n), 71, 202).round().astype(int)
    exang = rng.binomial(1, 0.33, n)
    oldpeak = np.clip(rng.gamma(1.2, 0.9, n), 0, 6.2).round(1)
    slope = rng.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47])

    # latent risk -> label (more risk => class 0 here is "less likely", class 1 = positive label)
    z = (
        -0.045 * (thalach - 150)
        + 0.7 * exang
        + 0.55 * oldpeak
        - 0.5 * (cp > 0)
        + 0.5 * sex
        - 0.6 * (slope == 2)
        + 0.02 * (age - 54)
        + rng.normal(0, 0.9, n)
    )
    target = (z < np.median(z)).astype(int)  # balanced classes, like the real set

    df = pd.DataFrame(dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                           fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                           oldpeak=oldpeak, slope=slope, target=target))
    df = df.drop_duplicates().reset_index(drop=True)
    extra = df.sample(N_DUP, replace=True, random_state=SEED)
    return pd.concat([df, extra]).sample(frac=1, random_state=SEED).reset_index(drop=True)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = make()
    out.to_csv(OUT, index=False)
    print(f"Wrote SYNTHETIC demo data: {OUT}  shape={out.shape}  unique={len(out.drop_duplicates())}")
