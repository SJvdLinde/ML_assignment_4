import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from models import LOWER_IS_BETTER, PRIMARY_METRIC, SCORING

FOLD_SCORES_PATH = "results/tables/fold_scores.csv"
TABLES_FOLDER = "results/tables"

MODEL_NAMES = ["knn", "tree", "forest", "boosting"]
ALPHA = 0.05


def load_scores(metric):

    fold_scores = pd.read_csv(FOLD_SCORES_PATH)
    scores = fold_scores.pivot(index="fold", columns="model", values=metric)
    scores = scores[MODEL_NAMES]

    return scores


def friedman_test(scores, lower_is_better):

    model_scores = [scores[model_name] for model_name in MODEL_NAMES]
    statistic, p_value = friedmanchisquare(*model_scores)

    ranks = scores.rank(axis=1, ascending=lower_is_better)
    mean_ranks = ranks.mean()

    return statistic, p_value, mean_ranks


def holm_correction(p_values):

    number_of_tests = len(p_values)


    order = sorted(range(number_of_tests), key=lambda i: p_values[i])

    adjusted = [0.0] * number_of_tests
    largest_so_far = 0.0

    for position, test_index in enumerate(order):
        multiplier = number_of_tests - position
        adjusted_p = min(1.0, multiplier * p_values[test_index])

        largest_so_far = max(largest_so_far, adjusted_p)
        adjusted[test_index] = largest_so_far

    return adjusted


def pairwise_tests(scores, lower_is_better):

    rows = []

    for first_index in range(len(MODEL_NAMES)):
        for second_index in range(first_index + 1, len(MODEL_NAMES)):
            first_model = MODEL_NAMES[first_index]
            second_model = MODEL_NAMES[second_index]

            first_scores = scores[first_model]
            second_scores = scores[second_model]

            statistic, p_value = wilcoxon(first_scores, second_scores)
            mean_difference = first_scores.mean() - second_scores.mean()

            if lower_is_better:
                first_wins = (first_scores < second_scores).sum()
            else:
                first_wins = (first_scores > second_scores).sum()

            rows.append({
                "model_a": first_model,
                "model_b": second_model,
                "mean_a_minus_b": mean_difference,
                "folds_a_better": int(first_wins),
                "statistic": statistic,
                "p_value": p_value,
            })

    pairwise = pd.DataFrame(rows)

    pairwise["holm_p_value"] = holm_correction(list(pairwise["p_value"]))
    pairwise["significant"] = pairwise["holm_p_value"] < ALPHA

    return pairwise


def wins_and_losses(pairwise, mean_ranks, lower_is_better):

    wins = {model_name: 0 for model_name in MODEL_NAMES}
    losses = {model_name: 0 for model_name in MODEL_NAMES}

    for _, row in pairwise.iterrows():
        if not row["significant"]:
            continue

        if lower_is_better:
            first_model_is_better = row["mean_a_minus_b"] < 0
        else:
            first_model_is_better = row["mean_a_minus_b"] > 0

        if first_model_is_better:
            winner = row["model_a"]
            loser = row["model_b"]
        else:
            winner = row["model_b"]
            loser = row["model_a"]

        wins[winner] += 1
        losses[loser] += 1

    rows = []
    for model_name in MODEL_NAMES:
        rows.append({
            "model": model_name,
            "wins": wins[model_name],
            "losses": losses[model_name],
            "wins_minus_losses": wins[model_name] - losses[model_name],
            "mean_rank": mean_ranks[model_name],
        })

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(["wins_minus_losses", "mean_rank"],
                                  ascending=[False, True])

    return ranking


def print_results(metric, statistic, p_value, pairwise, ranking):

    print(f"===== {metric} =====")
    print(f"Friedman test: statistic = {statistic:.3f}, p = {p_value:.3e}")

    if p_value < ALPHA:
        print(f"  reject H0 at alpha = {ALPHA}: at least one model differs\n")
    else:
        print(f"  cannot reject H0 at alpha = {ALPHA}\n")

    print("Pairwise Wilcoxon signed-rank tests with Holm correction:")
    print(pairwise.round(4).to_string(index=False), "\n")

    print("Ranking:")
    print(ranking.round(3).to_string(index=False), "\n")


if __name__ == "__main__":

    friedman_rows = []
    all_pairwise = []
    all_rankings = []

    for metric in SCORING:
        scores = load_scores(metric)
        lower_is_better = metric in LOWER_IS_BETTER

        statistic, p_value, mean_ranks = friedman_test(scores, lower_is_better)
        friedman_rows.append({
            "metric": metric,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < ALPHA,
        })

        pairwise = pairwise_tests(scores, lower_is_better)
        ranking = wins_and_losses(pairwise, mean_ranks, lower_is_better)

        pairwise.insert(0, "metric", metric)
        ranking.insert(0, "metric", metric)
        all_pairwise.append(pairwise)
        all_rankings.append(ranking)

        if metric == PRIMARY_METRIC:
            print_results(metric, statistic, p_value,
                          pairwise.drop(columns="metric"),
                          ranking.drop(columns="metric"))

    pd.DataFrame(friedman_rows).to_csv(f"{TABLES_FOLDER}/friedman.csv",
                                       index=False)
    pd.concat(all_pairwise).to_csv(f"{TABLES_FOLDER}/pairwise_tests.csv",
                                   index=False)
    pd.concat(all_rankings).to_csv(f"{TABLES_FOLDER}/ranking.csv", index=False)

    print(f"statistical tests written to {TABLES_FOLDER}/")
