import json
import sys
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from models import (KNN_GRID, PRIMARY_METRIC, RANDOM_STATE, SCORING, TREE_GRID,
                    build_knn, build_tree, tuning_cv)
from preprocessing import load_data


def run_grid_search(model_name, pipeline, grid, features, y, n_jobs):

    print(f"tuning {model_name} ...", flush=True)

    search = GridSearchCV(
        pipeline,
        grid,
        scoring=SCORING,
        refit=PRIMARY_METRIC,
        cv=tuning_cv(),
        n_jobs=n_jobs,
        verbose=1,
    )

    start = time.time()
    search.fit(features, y)
    minutes = (time.time() - start) / 60

    # save the score of every configuration
    all_scores = pd.DataFrame(search.cv_results_)
    score_columns = [c for c in all_scores.columns
                     if c.startswith(("param_", "mean_", "std_"))]
    all_scores = all_scores[score_columns].sort_values(
        f"mean_test_{PRIMARY_METRIC}", ascending=False)
    all_scores.to_csv(f"results/tables/{model_name}_grid.csv", index=False)

    print(f"  {len(all_scores)} configurations in {minutes:.1f} min")
    print(f"  best {PRIMARY_METRIC}: {search.best_score_:.4f}")
    print(f"  best params: {search.best_params_}\n")

    return search.best_params_

# The number of instances to use can be given as a command line argument:
#python src/tune.py 40000

if __name__ == "__main__":
    features, y = load_data()

    # tune on a subset of the data
    if len(sys.argv) > 1:
        sample_size = int(sys.argv[1])
        features, _, y, _ = train_test_split(features, y,
                                             train_size=sample_size,
                                             stratify=y,
                                             random_state=RANDOM_STATE)

    print(f"{len(features)} instances, {features.shape[1]} features\n", flush=True)

    best_params = {}

    # run the nearest neighbour grid search
    best_params["knn"] = run_grid_search("knn", build_knn(features), KNN_GRID,
                                         features, y, n_jobs=1)

    # run the classification tree grid search
    best_params["tree"] = run_grid_search("tree", build_tree(features), TREE_GRID,
                                          features, y, n_jobs=-1)

    with open("results/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("best parameters written to results/best_params.json")
