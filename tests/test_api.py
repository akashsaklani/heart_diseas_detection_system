import pytest

from heart_disease import config, data, predict
from heart_disease.model import build_random_forest
from make_synthetic_data import make
import joblib

SAMPLE = dict(age=45, sex=1, cp=0, trestbps=130, chol=230, fbs=0,
              restecg=1, thalach=150, exang=0, oldpeak=1.0, slope=1)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # train a throw-away model so tests never depend on files from a previous run
    df, _ = data.clean(make())
    X, y = data.split_xy(df)
    path = tmp_path_factory.mktemp("m") / "m.joblib"
    joblib.dump({"model": build_random_forest().fit(X, y), "features": config.FEATURES}, path)
    bundle = joblib.load(path)
    # point predict_one at the throw-away model (module-level lookup)
    monkeypatch_model = (bundle["model"], bundle["features"])
    original = predict.load_model
    predict.load_model = lambda: monkeypatch_model

    from app.app import app
    app.config["TESTING"] = True
    yield app.test_client()
    predict.load_model = original


def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"


def test_index_renders(client):
    assert b"Heart Disease Detection" in client.get("/").data


def test_predict_ok(client):
    r = client.post("/api/predict", json=SAMPLE)
    d = r.get_json()
    assert r.status_code == 200
    assert d["prediction"] in (0, 1)
    assert abs(d["probability_class_0"] + d["probability_class_1"] - 1) < 1e-3
    assert "not a medical diagnosis" in d["disclaimer"]


def test_predict_missing_field(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "age"}
    assert client.post("/api/predict", json=bad).status_code == 400


def test_predict_out_of_range(client):
    assert client.post("/api/predict", json={**SAMPLE, "age": 500}).status_code == 400


def test_predict_non_numeric(client):
    assert client.post("/api/predict", json={**SAMPLE, "chol": "abc"}).status_code == 400
