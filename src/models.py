from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer, matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import NOMINAL, knn_preprocessor, tree_preprocessor

RANDOM_STATE = 12
N_FOLDS = 5
N_REPEATS = 4

def build_knn(features):

    return Pipeline([
        ("prep", knn_preprocessor(features)),
        ("knn", KNeighborsClassifier(n_jobs=-1)),
    ])

def build_tree(features):

    return Pipeline([
        ("prep", tree_preprocessor(features)),
        ("tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])

def build_random_forest(features):

    forest = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    return Pipeline([("prep", tree_preprocessor(features)),("forest", forest)])

def nominal_column_positions(features):

    number_of_features = features.shape[1]
    number_of_nominal = len(NOMINAL)
    first_nominal_position = number_of_features - number_of_nominal

    positions = list(range(first_nominal_position, number_of_features))

    return positions

def build_gradient_boosting(features):

    nominal_positions = nominal_column_positions(features)

    boosting = HistGradientBoostingClassifier(categorical_features=nominal_positions,
                                              early_stopping=False,
                                              l2_regularization=1.0,
                                              random_state=RANDOM_STATE)

    return Pipeline([("prep", tree_preprocessor(features)), ("boosting", boosting)])


KNN_GRID = {
    "knn__n_neighbors": [1, 3, 5, 10, 20, 50],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}

TREE_GRID = {
    "tree__criterion": ["gini", "entropy"],
    "tree__max_depth": [None, 10, 20, 30],
    "tree__min_samples_leaf": [1, 5, 20],
    "tree__class_weight": [None, "balanced"],
    "tree__ccp_alpha": [0.0, 0.0001],
}

RANDOM_FOREST_GRID = {
    "forest__n_estimators": [50, 100, 200, 400],
    "forest__max_features": ["sqrt", "log2", 0.5, None],
    "forest__max_depth": [None, 20],
    "forest__max_samples": [0.25, 0.5, 1.0],
    "forest__class_weight": [None, "balanced"],
}

GRADIENT_BOOSTING_GRID = {
    "boosting__learning_rate": [0.05, 0.1, 0.2],
    "boosting__max_iter": [100, 300, 500],
    "boosting__max_leaf_nodes": [7, 15, 31, 63],
    "boosting__class_weight": [None, "balanced"],
}

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
