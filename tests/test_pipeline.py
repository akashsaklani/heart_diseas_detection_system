import pandas as pd
import pytest

from heart_disease import config, data
from heart_disease.model import build_random_forest
from make_synthetic_data import make


@pytest.fixture(scope="module")
def raw():
    return make()


def test_raw_shape(raw):
    assert raw.shape == (1025, 12)


def test_clean_removes_duplicates(raw):
    df, rep = data.clean(raw)
    assert rep.rows_clean == 302
    assert rep.duplicates_removed == 723
    assert rep.missing_total == 0
    assert not df.duplicated().any()


def test_split_matches_presentation(raw):
    df, _ = data.clean(raw)
    X_train, X_test, y_train, y_test = data.train_test(df)
    assert (len(X_train), len(X_test)) == (241, 61)


def test_missing_column_raises(raw):
    with pytest.raises(ValueError):
        data.clean(raw.drop(columns=["chol"]))


def test_rf_params():
    m = build_random_forest()
    assert m.n_estimators == 100 and m.random_state == 42


def test_model_learns(raw):
    df, _ = data.clean(raw)
    X_train, X_test, y_train, y_test = data.train_test(df)
    m = build_random_forest().fit(X_train, y_train)
    assert m.score(X_test, y_test) > 0.6  # clearly better than chance
    assert list(X_train.columns) == config.FEATURES
