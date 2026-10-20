import argparse
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from credit_default.experiments import RAW


def population_stability(risk, reference):
    risk = np.asarray(risk, dtype=float)
    if not len(risk) or not np.isfinite(risk).all() or ((risk < 0) | (risk > 1)).any():
        raise ValueError("Нужны вероятности от 0 до 1")
    edges = np.array([-np.inf, *reference["edges"][1:-1], np.inf])
    expected = np.asarray(reference["counts"], dtype=float)
    actual = np.histogram(risk, edges)[0].astype(float)
    # Сглаживание позволяет учитывать пустые интервалы.
    expected = (expected + 0.5) / (expected.sum() + 0.5 * len(expected))
    actual = (actual + 0.5) / (actual.sum() + 0.5 * len(actual))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.rows < 1 or not 1 <= args.batch_size <= 1000:
        parser.error("rows > 0, batch-size от 1 до 1000")
    frame = pd.read_csv("data/processed/test.csv")[RAW]
    frame = frame.sample(n=min(args.rows, len(frame)), random_state=42)
    reference = json.loads(Path("models/reference.json").read_text())
    risk = []
    with httpx.Client(base_url=args.url, timeout=30) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json()["model_sha256"] != reference["model_sha256"]:
            raise ValueError("API и фон PSI относятся к разным моделям")
        for start in range(0, len(frame), args.batch_size):
            batch = frame.iloc[start : start + args.batch_size].to_dict("records")
            response = client.post("/predict", json={"clients": batch})
            response.raise_for_status()
            risk.extend(p["probability"] for p in response.json()["predictions"])
    report = {
        "rows": len(risk),
        "probability_psi": population_stability(risk, reference),
        "mean_probability": float(np.mean(risk)),
    }
    output = Path("reports/monitoring.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
