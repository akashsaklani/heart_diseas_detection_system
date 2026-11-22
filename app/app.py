"""Flask web app + REST API for heart disease prediction.

Run:  python app/app.py        (dev)    ->  http://127.0.0.1:5000
      gunicorn -w 2 -b 0.0.0.0:5000 app.app:app    (production)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flask import Flask, jsonify, render_template, request  # noqa: E402

from heart_disease import config  # noqa: E402
from heart_disease.predict import ValidationError, predict_one  # noqa: E402

app = Flask(__name__)

FIELD_HELP = [
    {"name": f, "help": config.FEATURE_INFO[f],
     "min": config.FEATURE_RANGES[f][0], "max": config.FEATURE_RANGES[f][1]}
    for f in config.FEATURES
]


@app.get("/")
def index():
    return render_template("index.html", fields=FIELD_HELP, disclaimer=config.DISCLAIMER)


@app.get("/health")
def health():
    return jsonify(status="ok", model_present=config.MODEL_PATH.exists())


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if payload is None:  # allow normal form posts too
        payload = request.form.to_dict()
    try:
        return jsonify(predict_one(payload))
    except ValidationError as e:
        return jsonify(error=str(e)), 400
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
