"""Model definitions (workflow step 5)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config


def build_random_forest() -> RandomForestClassifier:
    """The model implemented in the project: 100 trees, random_state=42."""
    return RandomForestClassifier(**config.RF_PARAMS)


def candidate_models() -> dict:
    """All five algorithms studied in the internship, for comparison.

    Scale-sensitive models (LogReg, SVM, KNN) are wrapped with StandardScaler.
    """
    rs = config.RANDOM_STATE
    return {
        "Random Forest": build_random_forest(),
        "Logistic Regression": Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=rs),
        "SVM": Pipeline([("scale", StandardScaler()), ("clf", SVC(random_state=rs))]),
        "KNN": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
    }
