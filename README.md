# mlops_wine_project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An activity using MLOps best practices.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops_wine_module and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── mlops_wine_module   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_wine_module a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Reproducir los experimentos

### Prerrequisitos

- Linux/WSL con `make` instalado (`sudo apt install make`).
- Python 3.10 o superior.
- Conexión a internet la primera vez (el dataset Wine Quality se descarga de UCI).

Desde la raíz del proyecto, crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Las dependencias se instalan solas al correr `make train` (la regla `requirements`
ejecuta `pip install -r requirements.txt`). Para instalarlas manualmente:

```bash
make requirements
```

### Ejecutar los 5 experimentos

Correr en este orden, desde la raíz del proyecto y con el entorno activado.
Cada comando ejecuta `data` → `features` → `train` y registra un run en MLflow
(`mlflow.db`).

```bash
# 1. Baseline explícito
make train ARGS="--C 1.0 --max-iter 1000 --solver lbfgs --test-size 0.2 --seed 42 --cv-folds 5 --tracking-uri sqlite:///mlflow.db --run-name v1_baseline"

# 2. Regularización fuerte, clases balanceadas
make train ARGS="--C 0.01 --max-iter 500 --solver lbfgs --class-weight balanced --test-size 0.2 --seed 7 --cv-folds 5 --tracking-uri sqlite:///mlflow.db --run-name v2_c001_balanced"

# 3. Regularización débil, newton-cg
make train ARGS="--C 10 --max-iter 2000 --solver newton-cg --class-weight balanced --test-size 0.25 --seed 123 --cv-folds 10 --tracking-uri sqlite:///mlflow.db --run-name v3_c10_newton"

# 4. Solver sag, prueba más grande
make train ARGS="--C 0.5 --max-iter 3000 --solver sag --test-size 0.3 --seed 2024 --cv-folds 3 --tracking-uri sqlite:///mlflow.db --run-name v4_sag_test30"

# 5. Solver saga, prueba pequeña, muchos folds
make train ARGS="--C 2.0 --max-iter 5000 --solver saga --class-weight balanced --test-size 0.1 --seed 99 --cv-folds 8 --tracking-uri sqlite:///mlflow.db --run-name v5_saga_test10"
```

Notas:

- `--tracking-uri sqlite:///mlflow.db` es relativo al directorio actual, por eso hay que
  correr los comandos desde la raíz del proyecto.
- Cada corrida sobrescribe `models/model.pkl`; los 5 runs quedan guardados en MLflow.

### Ver los resultados en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir http://127.0.0.1:5000, entrar al experimento **TC5061 - Wine Quality baseline**,
seleccionar los 5 runs y pulsar **Compare**.

### Hacer predicciones con el último modelo

```bash
make predict
```

--------

