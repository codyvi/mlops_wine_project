from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from mlops_wine_module.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class WinePredictor:
    """Carga un modelo entrenado y genera predicciones."""

    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.model.predict_proba(X)
        out = pd.DataFrame(proba, columns=[f"proba_{c}" for c in self.model.classes_])
        out.insert(0, "prediction", self.model.predict(X))
        return out


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    logger.info("Performing inference for model...")
    preds = WinePredictor(model_path).predict(pd.read_csv(features_path))
    preds.to_csv(predictions_path, index=False)
    logger.success(f"Inference complete: {predictions_path}")


if __name__ == "__main__":
    app()
