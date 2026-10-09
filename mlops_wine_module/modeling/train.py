from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
import joblib
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mlops_wine_module.config import (
    CLASES,
    EXPERIMENT_NAME,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TRACKING_URI,
)
from mlops_wine_module.plots import ConfusionMatrixPlotter

app = typer.Typer()


@dataclass
class TrainConfig:
    """Hiperparámetros y ajustes de la corrida."""

    C: float = 1.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    class_weight: Optional[str] = None
    test_size: float = 0.2
    seed: int = 42
    cv_folds: int = 5


class WineTrainer:
    """Entrena y evalúa un baseline de regresión logística con seguimiento en MLflow."""

    def __init__(self, config: TrainConfig, tracking_uri: str = TRACKING_URI,
                 experiment_name: str = EXPERIMENT_NAME):
        self.config = config
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.plotter = ConfusionMatrixPlotter(CLASES)

    def build_model(self) -> Pipeline:
        c = self.config
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=c.C, max_iter=c.max_iter, solver=c.solver,
                class_weight=c.class_weight, random_state=c.seed,
            )),
        ])

    def split(self, X: pd.DataFrame, y: pd.Series):
        return train_test_split(
            X, y, test_size=self.config.test_size, stratify=y, random_state=self.config.seed)

    @staticmethod
    def evaluate(model, X_train, y_train, X_test, y_test):
        """Calcula métricas en prueba y devuelve también las predicciones."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "log_loss": log_loss(y_test, y_proba),
            "roc_auc_ovr": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
            "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
            "dummy_accuracy": accuracy_score(y_test, dummy.predict(X_test)),
        }
        return metrics, y_pred

    def cross_validate(self, X, y) -> dict:
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.seed)
        res = cross_validate(self.build_model(), X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        return {
            "cv_accuracy_mean": res["test_accuracy"].mean(),
            "cv_accuracy_std": res["test_accuracy"].std(ddof=1),
            "cv_f1_macro_mean": res["test_f1_macro"].mean(),
            "cv_f1_macro_std": res["test_f1_macro"].std(ddof=1),
        }

    def run(self, X: pd.DataFrame, y: pd.Series, run_name: str, model_path: Path,
            figures_dir: Path, test_dir: Path) -> dict:
        X_train, X_test, y_train, y_test = self.split(X, y)

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({
                **asdict(self.config),
                "model_type": "LogisticRegression",
                "n_features": X_train.shape[1],
                "n_train": len(X_train),
                "n_test": len(X_test),
            })

            model = self.build_model()
            model.fit(X_train, y_train)

            metrics, y_pred = self.evaluate(model, X_train, y_train, X_test, y_test)
            metrics.update(self.cross_validate(X, y))
            mlflow.log_metrics(metrics)

            # Artefactos: matriz de confusión y reporte de clasificación
            fig_path = self.plotter.save(y_test, y_pred, figures_dir / "matriz_confusion.png")
            mlflow.log_artifact(str(fig_path), artifact_path="evaluacion")
            mlflow.log_dict(
                classification_report(y_test, y_pred, target_names=CLASES, output_dict=True),
                "evaluacion/classification_report.json")

            mlflow.sklearn.log_model(
                sk_model=model,
                name="modelo_baseline",
                signature=infer_signature(X_train, model.predict(X_train)),
                input_example=X_train.head(3),
            )

            # Copias locales para predict.py / plots.py
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            test_dir.mkdir(parents=True, exist_ok=True)
            X_test.to_csv(test_dir / "test_features.csv", index=False)
            y_test.to_csv(test_dir / "test_labels.csv", index=False)

            logger.info(f"Run ID: {run.info.run_id}")
        return metrics


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    c: float = typer.Option(1.0, "--C", help="Inverso de la fuerza de regularización"),
    max_iter: int = typer.Option(1000, help="Iteraciones máximas del solver"),
    solver: str = typer.Option("lbfgs", help="lbfgs | newton-cg | sag | saga"),
    class_weight: Optional[str] = typer.Option(None, help="Usa 'balanced' para ponderar clases"),
    test_size: float = typer.Option(0.2, help="Proporción del conjunto de prueba"),
    seed: int = typer.Option(42, help="Semilla aleatoria"),
    cv_folds: int = typer.Option(5, help="Particiones de la validación cruzada"),
    tracking_uri: str = typer.Option(TRACKING_URI, help="URI de MLflow Tracking"),
    run_name: str = typer.Option("logreg_baseline", help="Nombre de la corrida"),
):
    logger.info("Training some model...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).iloc[:, 0]

    config = TrainConfig(c, max_iter, solver, class_weight, test_size, seed, cv_folds)
    trainer = WineTrainer(config, tracking_uri=tracking_uri)
    metrics = trainer.run(X, y, run_name, model_path, FIGURES_DIR, PROCESSED_DATA_DIR)

    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
