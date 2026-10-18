import json
import os
import sys
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from models import (GRADIENT_BOOSTING_GRID, PRIMARY_METRIC, RANDOM_FOREST_GRID,
                    RANDOM_STATE, SCORING, build_gradient_boosting,
                    build_random_forest, tuning_cv)
from preprocessing import load_data

BEST_PARAMS_PATH = "results/best_params.json"
TABLES_FOLDER = "results/tables"


def run_grid_search(model_name, pipeline, grid, features, target, n_jobs):

    print(f"tuning {model_name} ...", flush=True)

    search = GridSearchCV(
        pipeline,
        grid,
        scoring=SCORING,
        refit=PRIMARY_METRIC,
        cv=tuning_cv(),
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=1,
    )

    start = time.time()
    search.fit(features, target)
    minutes = (time.time() - start) / 60

    # save the score of every configuration
    all_scores = pd.DataFrame(search.cv_results_)
    score_columns = [c for c in all_scores.columns
                     if c.startswith(("param_", "mean_", "std_"))]
    all_scores = all_scores[score_columns].sort_values(
        f"mean_test_{PRIMARY_METRIC}", ascending=False)
    all_scores.to_csv(f"{TABLES_FOLDER}/{model_name}_grid.csv", index=False)

    print(f"  {len(all_scores)} configurations in {minutes:.1f} min")
    print(f"  best {PRIMARY_METRIC}: {search.best_score_:.4f}")
    print(f"  best params: {search.best_params_}\n")

    return search.best_params_


def load_best_params():

    # the assignment 2 configurations are already in this file, so read it
    # first and add to it, instead of overwriting it
    if not os.path.exists(BEST_PARAMS_PATH):
        return {}

    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)

    return best_params


def save_best_params(best_params):

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)

# The number of instances to use can be given as a command line argument:
#python src/tune.py 40000

if __name__ == "__main__":
    os.makedirs(TABLES_FOLDER, exist_ok=True)

    features, target = load_data()

    # tune on a subset of the data
    if len(sys.argv) > 1:
        sample_size = int(sys.argv[1])
        features, _, target, _ = train_test_split(features, target,
                                                  train_size=sample_size,
                                                  stratify=target,
                                                  random_state=RANDOM_STATE)

    print(f"{len(features)} instances, {features.shape[1]} features\n", flush=True)

    best_params = load_best_params()

    # both ensembles already use all the CPU cores on their own, so the grid
    # search itself runs one configuration at a time (n_jobs=1)

    # run the random forest grid search
    forest_pipeline = build_random_forest(features)
    best_params["forest"] = run_grid_search("forest", forest_pipeline,
                                            RANDOM_FOREST_GRID,
                                            features, target, n_jobs=1)
    save_best_params(best_params)

    # run the gradient boosting grid search
    boosting_pipeline = build_gradient_boosting(features)
    best_params["boosting"] = run_grid_search("boosting", boosting_pipeline,
                                              GRADIENT_BOOSTING_GRID,
                                              features, target, n_jobs=1)
    save_best_params(best_params)

    print(f"best parameters written to {BEST_PARAMS_PATH}")
