from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from credit_default.experiments import FEATURE_SETS, build_pipeline
from credit_default.features import add_features


def select_features(frame):
    return frame[FEATURE_SETS["base"]]


def build_model(seed=42, kind="cat"):
    # Признаки сохраняются вместе с моделью и применяются при каждом прогнозе.
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("columns", FunctionTransformer(select_features)),
            *build_pipeline(kind, "base", seed).steps,
        ]
    )
