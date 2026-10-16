import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

DATA_PATH = "data/networkTraffic.csv"

TARGET = "attack_cat"
NOMINAL = ["proto", "state", "service"]

REDUNDANT = ["sbytes", "sloss", "dbytes", "dloss", "dwin",
             "ct_dst_src_ltm", "ct_srv_dst", "ct_src_dport_ltm", "ct_ftp_cmd"]

MIN_FREQUENCY = 0.01


def load_data(path=DATA_PATH, dedup=True):

    df = pd.read_csv(path, na_values=["?"])
    df = df.drop(columns=["id"])

    if dedup:
        df = df.drop_duplicates()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def numeric_columns(X, drop_redundant):

    cols = [c for c in X.columns if c not in NOMINAL]
    if drop_redundant:
        cols = [c for c in cols if c not in REDUNDANT]

    return cols


def knn_preprocessor(X):

    numeric = numeric_columns(X, drop_redundant=True)

    numeric_steps = Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
    ])

    nominal_steps = Pipeline([
        ("fill", SimpleImputer(strategy="constant", fill_value="none")),
        ("encode", OneHotEncoder(
            min_frequency=MIN_FREQUENCY,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("nom", nominal_steps, NOMINAL),
    ])


def tree_preprocessor(X):

    numeric = numeric_columns(X, drop_redundant=False)

    nominal_steps = Pipeline([
        ("fill", SimpleImputer(strategy="constant", fill_value="none")),
        ("encode", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    return ColumnTransformer([
        ("num", "passthrough", numeric),
        ("nom", nominal_steps, NOMINAL),
    ])


if __name__ == "__main__":
    
    X, y = load_data()
    print("features:", X.shape, " target:", y.shape)
    print("classes:", sorted(y.unique()))

    knn = knn_preprocessor(X).fit_transform(X)
    print("after knn preprocessing:", knn.shape)

    tree = tree_preprocessor(X).fit_transform(X)
    print("after tree preprocessing:", tree.shape)
