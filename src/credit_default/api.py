import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StrictInt, create_model

from credit_default.data import BILLS, PAY, PAYMENTS
from credit_default.experiments import RAW

bounds = {
    "LIMIT_BAL": (1, None),
    "AGE": (18, 100),
    "SEX": (1, 2),
    "EDUCATION": (0, 6),
    "MARRIAGE": (0, 3),
}
bounds.update({c: (-2, 9) for c in PAY})
bounds.update({c: (0, None) for c in PAYMENTS})
bounds.update({c: (None, None) for c in BILLS})
Client = create_model(
    "Client",
    __config__=ConfigDict(extra="forbid"),
    **{c: (StrictInt, Field(ge=bounds[c][0], le=bounds[c][1])) for c in RAW},
)


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clients: list[Client] = Field(min_length=1, max_length=1000)


def create_app(model_path=None, threshold=0.5):
    path = Path(model_path or os.getenv("MODEL_PATH", "models/model.joblib"))

    @asynccontextmanager
    async def lifespan(app):
        # Загружаем один раз. Без файла модели сервис не запускается.
        app.state.model = joblib.load(path)
        app.state.threshold = getattr(app.state.model, "threshold_", threshold)
        app.state.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        yield

    app = FastAPI(title="Credit Default", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_sha256": app.state.model_sha256}

    @app.post("/predict")
    def predict(batch: Batch):
        frame = pd.DataFrame([c.model_dump() for c in batch.clients], columns=RAW)
        risk = app.state.model.predict_proba(frame)[:, 1]
        return {
            "predictions": [
                {"class": int(p >= app.state.threshold), "probability": float(p)}
                for p in risk
            ]
        }

    return app


app = create_app()
