import json
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_validate
from models import (N_FOLDS, SCORING, build_gradient_boosting,
                    build_knn, build_random_forest, build_tree, evaluation_cv,
                    tuning_cv)
from preprocessing import load_data

CLASS_NAMES = pd.read_csv("data/attack_category_map.csv", index_col=1).iloc[:, 0]

BEST_PARAMS_PATH = "results/best_params.json"
TABLES_FOLDER = "results/tables"
MODEL_BUILDERS = {"knn": build_knn, "tree": build_tree,
    "forest": build_random_forest, "boosting": build_gradient_boosting}


def evaluate(model_name, pipeline, features, target):

    print(f"evaluating {model_name} ...", flush=True)

    cv_results = cross_validate(pipeline, features, target, cv=evaluation_cv(),
                                scoring=SCORING, return_train_score=True)

    fold_scores = pd.DataFrame()
    fold_scores["model"] = [model_name]*len(cv_results["fit_time"])
    fold_scores["fold"] = range(1, len(fold_scores) + 1)

    for metric in SCORING:
        fold_scores[metric] = cv_results[f"test_{metric}"]
        fold_scores[f"train_{metric}"] = cv_results[f"train_{metric}"]


    fold_scores["fit_time"] = cv_results["fit_time"]
    fold_scores["predict_time"] = cv_results["score_time"]

    for metric in SCORING:
        test = cv_results[f"test_{metric}"]
        train = cv_results[f"train_{metric}"]
        print(f"  {metric:18s} test {test.mean():.4f} +- {test.std():.4f}"
              f"   train {train.mean():.4f}")

    print(f"  fit time {fold_scores['fit_time'].mean():.1f} s per fold")
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

def timing_table(scores_per_model, number_of_instances):

    test_fold_size = number_of_instances / N_FOLDS

    rows = []

    for model_name, fold_scores in scores_per_model.items():
        fit_time = fold_scores["fit_time"]
        predict_time = fold_scores["predict_time"]

        predict_ms_per_1000 = predict_time.mean() / test_fold_size * 1000 * 1000

        rows.append({
            "model": model_name, "fit_time_mean_s": fit_time.mean(),
            "fit_time_std_s": fit_time.std(), "predict_time_mean_s": predict_time.mean(),
            "predict_ms_per_1000_instances": predict_ms_per_1000})

    return pd.DataFrame(rows).round(4)


def per_class_report(model_name, pipeline, features, target):

    predictions = cross_val_predict(pipeline, features, target,
                                    cv=tuning_cv(), n_jobs=1)
    labels = sorted(target.unique())
    names = [CLASS_NAMES[i] for i in labels]

    report = classification_report(target, predictions, target_names=names,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.round(4).to_csv(
        f"{TABLES_FOLDER}/{model_name}_per_class.csv")

    matrix = pd.DataFrame(confusion_matrix(target, predictions, labels=labels),
                          index=names, columns=names)
    matrix.to_csv(f"{TABLES_FOLDER}/{model_name}_confusion_matrix.csv")


if __name__ == "__main__":
    os.makedirs(TABLES_FOLDER, exist_ok=True)

    features, target = load_data()
    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)

    pipelines = {}
    for model_name, build_model in MODEL_BUILDERS.items():
        pipeline = build_model(features)
        pipeline.set_params(**best_params[model_name])
        pipelines[model_name] = pipeline
        print(f"{model_name}: {best_params[model_name]}")
    print()

    scores_per_model, results_per_model = {}, {}

    for model_name, pipeline in pipelines.items():
        fold_scores, cv_results = evaluate(model_name, pipeline, features, target)
        scores_per_model[model_name] = fold_scores
        results_per_model[model_name] = cv_results


        all_folds = pd.concat(scores_per_model.values(), ignore_index=True)
        all_folds.to_csv(f"{TABLES_FOLDER}/fold_scores.csv", index=False)

    summary = summary_table(results_per_model)
    summary.to_csv(f"{TABLES_FOLDER}/final_results.csv", index=False)
    print(summary.to_string(index=False), "\n")

    timing = timing_table(scores_per_model, len(target))
    timing.to_csv(f"{TABLES_FOLDER}/timing.csv", index=False)
    print(timing.to_string(index=False), "\n")


    # per-class scores and confusion matrices
    for model_name, pipeline in pipelines.items():
        print(f"per-class results for {model_name} ...", flush=True)
        per_class_report(model_name, pipeline, features, target)
