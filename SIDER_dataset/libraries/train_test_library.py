import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.tree import DecisionTreeClassifier
from typing import Literal
import logging
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from sklearn.inspection import permutation_importance


def train_n_test(
        alg: Literal["DT", "RF"],
        training_set: pd.DataFrame,
        test_set: pd.DataFrame,
        labels: list[str],
        cv: bool,
        scoring: list[str] = ["precision", "recall", "f1", "roc_auc"],
):
    feature_columns = training_set.columns.drop(labels)
    label_columns = labels

    # prepare training/testing datasets
    X_train = training_set[feature_columns]
    y_train = training_set[label_columns]

    if alg == "DT":
        model = DecisionTreeClassifier(random_state=42)
    elif alg == "RF":
        model = RandomForestClassifier(
            n_estimators=500, max_features="sqrt", random_state=42
        )
    if len(labels) > 1:
        model = MultiOutputClassifier(model)
    model.fit(X_train, y_train)
    importances = get_mdi_importance_df(X_train, model)

    if cv:
        if len(labels) > 1:
            raise ValueError("CV not implemented for multi-label")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring)
        return model, get_results_from_cv_scores(cv_scores, scoring), importances

    elif test_set is not None:
        X_test = test_set[feature_columns]
        y_test = test_set[label_columns]
        # Predict
        y_pred = model.predict(X_test)
        if len(labels) == 1:
            results = get_sl_performance_from_test_set(
                X_test, y_test, y_pred, model, scoring
            )
        elif len(labels) > 1:
            results = get_ml_performance_from_test_set(
                labels, X_test, y_test, y_pred, model, scoring
            )
        return model, results, importances


def get_sl_performance_from_test_set(
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred: np.ndarray,
        model: DecisionTreeClassifier | RandomForestClassifier,
):
    results = {}
    results["accuracy"] = accuracy_score(y_test, y_pred)
    results["precision"] = precision_score(y_test, y_pred)
    results["recall"] = recall_score(y_test, y_pred)
    results["f1"] = f1_score(y_test, y_pred)

    y_proba = model.predict_proba(X_test)[:, 1]
    results["roc_auc"] = roc_auc_score(y_test, y_proba)
    return results


def get_ml_performance_from_test_set(
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred: np.ndarray,
        model: MultiOutputClassifier,
):
    results = {}
    results["accuracy"] = accuracy_score(y_test, y_pred, average="macro")
    results["precision"] = precision_score(y_test, y_pred, average="macro")
    results["recall"] = recall_score(y_test, y_pred, average="macro")
    results["f1"] = f1_score(y_test, y_pred, average="macro")

    y_proba_list = model.predict_proba(X_test)
    y_proba = np.column_stack([proba[:, 1] for proba in y_proba_list])
    results["roc_auc"] = roc_auc_score(y_test, y_proba, average="macro")
    return results


def get_perm_importance_df(model, X_test, y_test, features):
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_importances = pd.DataFrame(
        perm_result.importances_mean, index=features, columns=["MDA"]
    )

    return perm_importances


def get_mdi_importance_df(
        X_train: pd.DataFrame,
        model: DecisionTreeClassifier | RandomForestClassifier | MultiOutputClassifier,
):
    if isinstance(model, MultiOutputClassifier):
        importances_list = [est.feature_importances_ for est in model.estimators_]
        avg_importances = np.mean(importances_list, axis=0)
    else:
        # Single-output model
        avg_importances = model.feature_importances_

    feature_importance_df = pd.DataFrame(
        avg_importances,
        index=X_train.columns,
        columns=["MDI"],
    )
    return feature_importance_df


def get_results_from_cv_scores(cv_scores, scoring: list[str]):
    results = {}
    for metric in scoring:
        results[metric] = cv_scores[f"test_{metric}"].mean()
    return results


def highlight_max(s):
    is_max = s == s.max()
    return ["background-color: green" if v else "" for v in is_max]


def print_split_info(title, dataset, label):
    print(f"{title}:")
    feature_columns = dataset.columns.drop(label)
    label_column = [label]
    negative_examples = len(dataset[dataset[label] == 0])
    positive_examples = len(dataset[dataset[label] == 1])
    print(f"negative examples: {negative_examples}")
    print(f"positive examples: {positive_examples}")
    print(f"feature_columns_size: {len(feature_columns)}")
    print(f"feature_columns: {feature_columns}")
    print(f"label_columns_size: {len(label_column)}")
    print(f"label_columns_size: {label_column}")
    print("\n")


def get_logger(logging_path: str):
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=logging_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )
    return logging.getLogger()


def get_test_df(
        num_of_rows: int,
        num_of_features: int,
        num_of_labels: int,
        random_state: int,
):
    np.random.seed(random_state)

    half = num_of_features // 2

    # Binary
    binary_feature_data = np.random.randint(0, 2, size=(num_of_rows, half))
    binary_feature_columns = [f"b_feature_{i}" for i in range(1, half + 1)]
    binary_feature_data[:, 0] = 1  # set to the same value across all rows
    # all 0s, first row is 1
    binary_feature_data[:, 1] = 0
    binary_feature_data[0, 1] = 1

    # Probability scores (0-1000)
    prob_feature_data = np.random.randint(0, 1001, size=(num_of_rows, half))
    prob_feature_columns = [f"p_feature_{i}" for i in range(1, half + 1)]
    prob_feature_data[:, 0] = 699  # set to the same value across all rows
    # all 0s, first row is 1
    prob_feature_data[:, 1] = 0
    prob_feature_data[0, 1] = 800

    feature_data = np.hstack((binary_feature_data, prob_feature_data))
    feature_columns = binary_feature_columns + prob_feature_columns
    feature_df = pd.DataFrame(feature_data, columns=feature_columns)

    # Labels
    label_data = np.random.randint(0, 2, size=(num_of_rows, num_of_labels))
    label_columns = [f"label_{i}" for i in range(1, num_of_labels + 1)]
    label_df = pd.DataFrame(label_data, columns=label_columns)

    # Add duplicate rows across features only - labels would have different values across rows
    inconsistent_rows = 20
    duplicate_feature_row = feature_df.iloc[[0]]
    duplicate_feature_rows = pd.concat(
        [duplicate_feature_row] * inconsistent_rows, ignore_index=True
    )
    new_label_data = np.random.randint(
        0, 2, size=(inconsistent_rows, label_df.shape[1])
    )
    new_label_rows = pd.DataFrame(new_label_data, columns=label_df.columns)
    feature_df.iloc[-inconsistent_rows:] = duplicate_feature_rows.values
    label_df.iloc[-inconsistent_rows:] = new_label_rows.values
    df = pd.concat([feature_df, label_df], axis=1)

    # Add duplicate rows across both features and labels
    duplicate_rows = 10
    duplicate_row = df.iloc[[0]]
    duplicates = pd.concat([duplicate_row] * duplicate_rows, ignore_index=True)
    df.iloc[-duplicate_rows:] = duplicates.values

    return (
        df,
        label_columns,
        feature_columns,
        binary_feature_columns,
        prob_feature_columns,
    )


def train(
        alg: Literal["DT", "RF"],
        training_set: pd.DataFrame,
        label_cols: list[str],
        output=True,
):
    if output:
        print("\ntrain")
    feature_cols = training_set.columns.drop(label_cols)
    X_train = training_set[feature_cols]
    y_train = training_set[label_cols]
    if output:
        print(
            f" Training with: algorithm: {alg}    instances: {len(X_train)}   features: {len(feature_cols)}   labels: {len(label_cols)} {label_cols}"
        )

    if alg == "DT":
        model = DecisionTreeClassifier(random_state=42)
    elif alg == "RF":
        model = RandomForestClassifier(
            n_estimators=500, max_features="sqrt", random_state=42
        )
    if len(label_cols) > 1:
        if output:
            print(" Training with MultiOutputClassifier")
        model = MultiOutputClassifier(model)
    else:
        y_train = y_train.values.ravel()

    if output:
        print(" Training...")
    model.fit(X_train, y_train)
    return model


def test(
        model: DecisionTreeClassifier | RandomForestClassifier | MultiOutputClassifier,
        test_dataset: pd.DataFrame,
        label_cols: list[str],
        output=True,
):
    if output:
        print("\ntest")
    feature_cols = test_dataset.columns.drop(label_cols)
    X_test = test_dataset[feature_cols]
    y_test = test_dataset[label_cols]
    if output:
        print(f" Testing with:  instances: {len(X_test)}")
    y_pred = model.predict(X_test)
    if isinstance(model, MultiOutputClassifier):
        results = get_ml_performance_from_test_set(X_test, y_test, y_pred, model)
    else:
        results = get_sl_performance_from_test_set(X_test, y_test, y_pred, model)
    return pd.DataFrame([results], index=label_cols)


def get_mean_cv_results(all_labels_cv_results_df: pd.DataFrame):
    group_cols = [
        "index",
        "ranking_criterion",
        "training_algorithm",
        "eval_criterion",
        "include_original_features",
    ]
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    grouped_means = (
        all_labels_cv_results_df.groupby(group_cols)[metric_cols].mean().reset_index()
    )
    return grouped_means


def get_macro_results(mean_labels_results: list[pd.DataFrame]):
    mean_labels_results_df = pd.concat(mean_labels_results).reset_index()
    mean_labels_results_df.index.name = "label"
    mean_row = mean_labels_results_df.mean(numeric_only=True)
    mean_row_df = pd.DataFrame([mean_row])
    mean_row_df["label"] = "Macro"
    mean_labels_results_df = pd.concat(
        [mean_labels_results_df, mean_row_df], ignore_index=True
    )
    return mean_labels_results_df
