import numpy as np
import pandas as pd

DEBUG = False


# --- Variance functions ---
def node_variance(Y: pd.DataFrame):
    """Compute multi-label variance for a node (Y: n x m DataFrame of binary or numeric labels)."""
    if Y.shape[0] == 0:
        if DEBUG:
            print("Empty node -> variance = 0")
        return 0.0

    # Ensure Y is a DataFrame
    if not isinstance(Y, pd.DataFrame):
        Y = pd.DataFrame(Y)

    p = Y.mean(axis=0)
    if DEBUG:
        print(f"  Label means (probabilities of 1): {p.to_numpy()}")

    var_per_label = p * (1 - p)
    if DEBUG:
        print(f"  Per-label variances: {var_per_label.to_numpy()}")

    total_var = var_per_label.sum()
    if DEBUG:
        print(f"  Total variance of node: {total_var}")

    return total_var


# --- Feature scoring ---
def best_variance_reduction_for_feature(x: pd.DataFrame, Y: pd.DataFrame, min_samples_leaf=2):
    """
    Compute best variance reduction for one numeric feature at the root level.
    x: pandas DataFrame with a single column (n x 1)
    Y: pandas DataFrame (n x m)
    """
    # Validate and normalize inputs
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)

    if not isinstance(Y, pd.DataFrame):
        Y = pd.DataFrame(Y)

    if x.shape[1] != 1:
        raise ValueError("Input x must be a single-column DataFrame.")

    feature_name = x.columns[0]
    x_values = x[feature_name].values
    n = len(x_values)

    if DEBUG:
        print(f"\nEvaluating feature '{feature_name}' with {n} samples")

    if n == 0:
        return 0.0, None

    # Sort feature and align labels
    order = np.argsort(x_values)
    xs = x_values[order]
    Ys = Y.iloc[order]

    if DEBUG:
        print(f"  Sorted feature values (first 10): {xs[:10]}")

    parent_var = node_variance(Ys)
    if DEBUG:
        print(f"Parent variance: {parent_var}")

    best_reduction = 0.0
    best_threshold = None

    # Candidate thresholds: midpoints between consecutive distinct values
    for i in range(min_samples_leaf, n - min_samples_leaf):
        if xs[i] == xs[i - 1]:
            continue

        thr = (xs[i] + xs[i - 1]) / 2
        left, right = Ys.iloc[:i], Ys.iloc[i:]

        if DEBUG:
            print(f"    Candidate threshold: {thr:.3f}")
            print(f"      Left size: {len(left)}, Right size: {len(right)}")

        left_var = node_variance(left)
        right_var = node_variance(right)

        weighted_var = (len(left) / n) * left_var + (len(right) / n) * right_var
        var_reduct = parent_var - weighted_var

        if DEBUG:
            print(f"      Weighted child variance: {weighted_var:.4f}, Reduction: {var_reduct:.4f}")

        if var_reduct > best_reduction:
            best_reduction = var_reduct
            best_threshold = thr
            if DEBUG:
                print(f"      -> New best split found at threshold {thr:.3f} with reduction {var_reduct:.4f}")

    return best_reduction, best_threshold


# --- Main scoring function ---
def feature_variance_reduction_scores(X: pd.DataFrame, Y: pd.DataFrame, min_samples_leaf=2):
    """
    Compute best variance reduction score for each feature (root-level only).
    X: pandas DataFrame of numeric features (n x p)
    Y: pandas DataFrame of labels (n x m)
    """
    # Ensure DataFrames
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(Y, pd.DataFrame):
        Y = pd.DataFrame(Y)

    results = []
    for col in X.columns:
        if DEBUG:
            print(f"\n========== Evaluating feature: {col} ==========")
        score, thr = best_variance_reduction_for_feature(X[[col]], Y, min_samples_leaf)
        if DEBUG:
            print(f"Best score for feature {col}: {score:.4f} at threshold {thr}")
        results.append({
            "best_variance_reduction": score,
            "best_threshold": thr
        })

    return pd.DataFrame(results, index=X.columns).sort_values("best_variance_reduction",
                                                              ascending=False)
