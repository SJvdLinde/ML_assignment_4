# ML Assignment 4

Ensemble learning for predicting cyberattack.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `networkTraffic.csv` in `data/`

## Run

Run from the project root:

```bash
python src/tune.py 40000            # tune the ensemble
python src/evaluate.py              # evaluate all four models
python src/statistical_tests.py     # Friedman and Holm-corrected Wilcoxon tests
python src/plots.py                 # figures
```

Results are written to `results/`.
