from sklearn.metrics import cohen_kappa_score, make_scorer, matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import knn_preprocessor, tree_preprocessor

RANDOM_STATE = 42
N_FOLDS = 5
N_REPEATS = 4


def build_knn(X):

    return Pipeline([
        ("prep", knn_preprocessor(X)),
        ("knn", KNeighborsClassifier(n_jobs=-1)),
    ])


def build_tree(X):

    return Pipeline([
        ("prep", tree_preprocessor(X)),
        ("tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])


# knn tuning grid search parameters
KNN_GRID = {
    "knn__n_neighbors": [1, 3, 5, 10, 20, 50],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}

# tree tuning grid search parameters
TREE_GRID = {
    "tree__criterion": ["gini", "entropy"],
    "tree__max_depth": [None, 10, 20, 30],
    "tree__min_samples_leaf": [1, 5, 20],
    "tree__class_weight": [None, "balanced"],
    "tree__ccp_alpha": [0.0, 0.0001],
}

# scoring on multiple metrics
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "mcc": make_scorer(matthews_corrcoef),
    "kappa": make_scorer(cohen_kappa_score),
}

PRIMARY_METRIC = "f1_macro"


def tuning_cv():
    """Cross-validation used during the grid search."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE)


def evaluation_cv():
    """Cross-validation used for the final performance estimates."""
    return RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                   random_state=RANDOM_STATE)
