from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mlops_wine_module.config import PROCESSED_DATA_DIR

app = typer.Typer()


class FeatureBuilder:
    """Separa el dataset procesado en matriz de features y etiquetas."""

    def __init__(self, target_col: str = "target"):
        self.target_col = target_col

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df.drop(columns=self.target_col), df[self.target_col].astype(int)

    def build(self, input_path: Path, features_path: Path, labels_path: Path) -> None:
        X, y = self.split(pd.read_csv(input_path))
        features_path.parent.mkdir(parents=True, exist_ok=True)
        X.to_csv(features_path, index=False)
        y.to_csv(labels_path, index=False)


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
):
    logger.info("Generating features from dataset...")
    FeatureBuilder().build(input_path, features_path, labels_path)
    logger.success(f"Features generation complete: {features_path}, {labels_path}")


if __name__ == "__main__":
    app()
