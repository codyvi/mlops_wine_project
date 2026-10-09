from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")  # sin interfaz gráfica: corre en terminal/CI
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import typer

from mlops_wine_module.config import CLASES, FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class ConfusionMatrixPlotter:
    """Genera y guarda la matriz de confusión del conjunto de prueba."""

    def __init__(self, labels: list[str] = CLASES):
        self.labels = labels

    def plot(self, y_true, y_pred) -> Figure:
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_true, y_pred), display_labels=self.labels
        ).plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title("Matriz de confusión (conjunto de prueba)")
        ax.grid(False)
        fig.tight_layout()
        return fig

    def save(self, y_true, y_pred, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = self.plot(y_true, y_pred)
        fig.savefig(output_path)
        plt.close(fig)
        return output_path


@app.command()
def main(
    labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    output_path: Path = FIGURES_DIR / "matriz_confusion.png",
):
    """Grafica la matriz de confusión a partir de etiquetas y predicciones guardadas."""
    import pandas as pd

    logger.info("Generating plot from data...")
    y_true = pd.read_csv(labels_path).iloc[:, 0]
    y_pred = pd.read_csv(predictions_path)["prediction"]
    ConfusionMatrixPlotter().save(y_true, y_pred, output_path)
    logger.success(f"Plot generation complete: {output_path}")


if __name__ == "__main__":
    app()
