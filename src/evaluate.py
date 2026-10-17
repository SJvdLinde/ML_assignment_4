import json

import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_validate
from models import (PRIMARY_METRIC, SCORING, build_knn, build_tree,
                    evaluation_cv, tuning_cv)
from preprocessing import load_data

CLASS_NAMES = pd.read_csv("data/attack_category_map.csv", index_col=1).iloc[:, 0]


def evaluate(model_name, pipeline, features, y):

    print(f"evaluating {model_name} ...", flush=True)

    cv_results = cross_validate(pipeline, features, y, cv=evaluation_cv(),
                                scoring=SCORING, return_train_score=True)

    fold_scores = pd.DataFrame({m: cv_results[f"test_{m}"] for m in SCORING})
    fold_scores["model"] = model_name
    fold_scores["fold"] = range(1, len(fold_scores) + 1)

    for metric in SCORING:
        test = cv_results[f"test_{metric}"]
        train = cv_results[f"train_{metric}"]
        print(f"  {metric:18s} test {test.mean():.4f} +- {test.std():.4f}"
              f"   train {train.mean():.4f}")
        
    print()

    return fold_scores, cv_results

def summary_table(results_per_model):

    rows = []

    for model_name, cv_results in results_per_model.items():
        for metric in SCORING:
            rows.append({
                "model": model_name,
                "metric": metric,
                "train_mean": cv_results[f"train_{metric}"].mean(),
                "train_std": cv_results[f"train_{metric}"].std(),
                "test_mean": cv_results[f"test_{metric}"].mean(),
                "test_std": cv_results[f"test_{metric}"].std(),
            })

    return pd.DataFrame(rows).round(4)


def compare(knn_scores, tree_scores):

    knn_f1 = knn_scores[PRIMARY_METRIC].to_numpy()
    tree_f1 = tree_scores[PRIMARY_METRIC].to_numpy()

    statistic, p_value = wilcoxon(knn_f1, tree_f1)
    better = "tree" if tree_f1.mean() > knn_f1.mean() else "knn"

    print("Wilcoxon signed-rank test on paired " + PRIMARY_METRIC)
    print(f"  knn  {knn_f1.mean():.4f} +- {knn_f1.std():.4f}")
    print(f"  tree {tree_f1.mean():.4f} +- {tree_f1.std():.4f}")
    print(f"  n = {len(knn_f1)} paired folds, "
          f"statistic = {statistic}, p = {p_value:.3e}")
    if p_value < 0.05:
        print(f"  reject H0 at alpha = 0.05: {better} performs better\n")
    else:
        print("  cannot reject H0 at alpha = 0.05\n")

    return {"statistic": float(statistic), "p_value": float(p_value),
            "better": better}


def per_class_report(model_name, pipeline, features, y):

    predictions = cross_val_predict(pipeline, features, y, cv=tuning_cv(), n_jobs=1)
    labels = sorted(y.unique())
    names = [CLASS_NAMES[i] for i in labels]

    report = classification_report(y, predictions, target_names=names,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.round(4).to_csv(
        f"results/tables/{model_name}_per_class.csv")

    matrix = pd.DataFrame(confusion_matrix(y, predictions, labels=labels),
                          index=names, columns=names)
    matrix.to_csv(f"results/tables/{model_name}_confusion_matrix.csv")


if __name__ == "__main__":
    features, y = load_data()
    with open("results/best_params.json") as f:
        best_params = json.load(f)

    # rebuild each model using the configuration found by the grid search
    pipelines = {"knn": build_knn(features), "tree": build_tree(features)}
    for model_name, pipeline in pipelines.items():
        pipeline.set_params(**best_params[model_name])
        print(f"{model_name}: {best_params[model_name]}")
    print()

    # score both models with the cross-validation
    scores_per_model, results_per_model = {}, {}
    for model_name, pipeline in pipelines.items():
        fold_scores, cv_results = evaluate(model_name, pipeline, features, y)
        scores_per_model[model_name] = fold_scores
        results_per_model[model_name] = cv_results

    summary = summary_table(results_per_model)
    summary.to_csv("results/tables/final_results.csv", index=False)
    print(summary.to_string(index=False), "\n")

    all_folds = pd.concat(scores_per_model.values(), ignore_index=True)
    all_folds.round(4).to_csv("results/tables/fold_scores.csv", index=False)

    # test whether the difference between the models is significant
    comparison = compare(scores_per_model["knn"], scores_per_model["tree"])
    with open("results/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # per-class scores and confusion matrices
    for model_name, pipeline in pipelines.items():
        print(f"per-class results for {model_name} ...", flush=True)
        per_class_report(model_name, pipeline, features, y)

    print("tables written to results/tables/")
