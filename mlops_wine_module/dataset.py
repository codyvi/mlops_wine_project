from pathlib import Path

from loguru import logger
import pandas as pd
from ucimlrepo import fetch_ucirepo
import typer

from mlops_wine_module.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, UCI_DATASET_ID

app = typer.Typer()


class WineDataset:
    """Descarga y preprocesa Wine Quality, dejando el objetivo en 3 clases."""

    def __init__(self, uci_id: int = UCI_DATASET_ID):
        self.uci_id = uci_id

    def download(self) -> pd.DataFrame:
        """Descarga el dataset original (features + color + quality)."""
        original = fetch_ucirepo(id=self.uci_id).data.original
        return original.copy()

    @staticmethod
    def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
        """Construye features y target, convierte a numérico, imputa con la mediana
        y elimina duplicados exactos.

        Los atípicos IQR se conservan por ser químicamente plausibles.
        """
        df = raw.drop(columns=["quality", "color"])
        df["tipo_tinto"] = (raw["color"] == "red").astype(int)
        # Calidad 3-5 -> 0 (baja), 6 -> 1 (media), 7-9 -> 2 (alta)
        df["target"] = pd.cut(raw["quality"], bins=[0, 5, 6, 10], labels=False)

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.fillna(df.median())
        df = df.drop_duplicates().reset_index(drop=True)
        df["target"] = df["target"].astype(int)
        return df

    def build(self, raw_path: Path, output_path: Path) -> pd.DataFrame:
        """Usa el CSV crudo si existe; si no, lo descarga. Guarda el procesado."""
        if raw_path.exists():
            logger.info(f"Leyendo datos crudos de {raw_path}")
            raw = pd.read_csv(raw_path)
        else:
            logger.info(f"Descargando Wine Quality (UCI id={self.uci_id})...")
            raw = self.download()
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw.to_csv(raw_path, index=False)

        df = self.preprocess(raw)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df


@app.command()
def main(
    raw_path: Path = RAW_DATA_DIR / "wine_quality.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    logger.info("Processing dataset...")
    df = WineDataset().build(raw_path, output_path)
    logger.success(f"Processing dataset complete: {df.shape} -> {output_path}")


if __name__ == "__main__":
    app()
