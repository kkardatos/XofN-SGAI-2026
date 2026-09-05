import pandas as pd

from SIDER_dataset.libraries.PCT_library import run_PCT, remove_PCT_files
from SIDER_dataset.libraries.feature_evaluation_methods import feature_variance_reduction_scores, \
    best_variance_reduction_for_feature
from SIDER_dataset.libraries.prepare_dataset_library import *
from SIDER_dataset.libraries.train_test_library import *
import json
from pathlib import Path
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import Counter
from sklearn.model_selection import cross_val_score
import os
import shutil
import time
from sklearn.preprocessing import StandardScaler

DEBUG = False


def group_features(
        dataset: pd.DataFrame,
        labels: list[str],
        groups: list[list[str]],
        include_original: bool,
        verbose=False
):
    original_features = get_features(dataset, labels)
    features_in_groups = [item for sublist in groups for item in sublist]
    ungrouped_features = [item for item in original_features if item not in features_in_groups]
    single_len_group_features = [g for g in groups if len(g) == 1]
    ungrouped_features.extend([g[0] for g in single_len_group_features])
    group_features = [g for g in groups if len(g) > 1]

    grouped_cols = {}
    for group in group_features:
        # ✅ Use full names, not stripped
        f_group_name = "_".join(group)
        # ✅ Preserve same index alignment
        grouped_cols[f_group_name] = dataset[group].sum(axis=1)

    grouped_dataset = pd.DataFrame(grouped_cols, index=dataset.index)
    ungrouped_dataset = dataset[ungrouped_features]
    label_dataset = dataset[labels]

    if include_original:
        dataset = pd.concat(
            [dataset[original_features], grouped_dataset, label_dataset],
            axis=1,
        )
    else:
        dataset = pd.concat([grouped_dataset, ungrouped_dataset, label_dataset], axis=1)

    return dataset


def get_cache_results(cache_results_path: str):
    if Path(cache_results_path).exists():
        with open(cache_results_path, "r") as f:
            cache_results = json.load(f)
    else:
        cache_results = {}
    return cache_results


def calculate_mdi_multi_rf(
        learn_set: pd.DataFrame,
        labels: list[str],
        random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute Mean Decrease in Impurity (MDI) per label and averaged across labels
    using Random Forest classification.

    Args:
        learn_set: Training DataFrame (features + labels).
        labels: List of target columns.
        random_state: Random state for reproducibility.

    Returns:
        pd.DataFrame with MDI per feature, per label, and average across labels.
    """
    features = [col for col in learn_set.columns if col not in labels]
    all_mdi = pd.DataFrame(index=features)

    for label in labels:
        rf = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(learn_set[features], learn_set[label])
        all_mdi[label] = rf.feature_importances_

    # Add mean importance across labels
    all_mdi["MDI_mean"] = all_mdi.mean(axis=1)

    # Sort by average score
    all_mdi = all_mdi.sort_values("MDI_mean", ascending=False)

    return all_mdi


def calculate_feature_perfs(learn_set: pd.DataFrame,
                            val_set: pd.DataFrame,
                            label: str,
                            measures: list[str] = ["roc_auc", "MDI"],
                            ) -> pd.DataFrame:
    single_f_results = []
    features = [col for col in learn_set.columns if col != label]
    for f in features:
        single_f_learn_set = learn_set[[f, label]]
        single_f_val_set = val_set[[f, label]]
        model = train("DT", single_f_learn_set, [label], False)
        performance_df = test(model, single_f_val_set, [label], False)
        performance_df.index = [f]
        if DEBUG:
            print(performance_df.iloc[[0]])
        single_f_results.append(performance_df)

    single_f_results_df = pd.concat(single_f_results)
    training_set = pd.concat([learn_set, val_set])
    model = train("DT", training_set, [label])

    # model, X_test, y_test, features
    mdi_importances = get_mdi_importance_df(training_set[features], model)
    perm_importances = get_perm_importance_df(
        model, val_set[features], val_set[label], features
    )

    importances = pd.merge(
        mdi_importances,
        perm_importances,
        left_index=True,
        right_index=True,
    )

    feature_rankings = pd.merge(
        single_f_results_df,
        importances,
        left_index=True,
        right_index=True,
    )

    feature_rankings["norm_roc_auc"] = (
            feature_rankings["roc_auc"] / feature_rankings["roc_auc"].sum()
    )
    feature_rankings["norm_MDA"] = (
            feature_rankings["MDA"] / feature_rankings["MDA"].sum()
    )
    feature_rankings["AUROC_MDI_MDA"] = feature_rankings[
        ["norm_roc_auc", "MDI", "norm_MDA"]
    ].mean(axis=1)

    feature_rankings["label"] = label
    return feature_rankings


def generate_feature_rankings(
        feature_rankings: pd.DataFrame,
        ranking_criterion: str,
):
    print_message = f"\ncalculate_feature_perfs -> Generating rankings based on ranking_criterion:{ranking_criterion}."
    print(print_message)
    if ranking_criterion == "random":
        feature_rankings = feature_rankings.sample(frac=1, random_state=42)
    else:
        feature_rankings = feature_rankings.sort_values(
            by=ranking_criterion, ascending=False
        )
    if DEBUG:
        print(feature_rankings.to_string())
    return feature_rankings


def generate_XofN_list(
        alg: Literal["DT", "RF"],
        learn_set: pd.DataFrame,
        val_set: pd.DataFrame,
        rankings: pd.DataFrame,
        eval_criterion: str,
        max_n: int,
        label: str,
        logger: logging.Logger,
):
    print(
        f"\ngenerate_XofN_list -> Generating groupings based on eval_criterion:{eval_criterion}."
    )
    X_of_N_list = []
    ranked_features = list(rankings.index)  # Ensure consistent indexing
    for i, f_i in enumerate(
            tqdm(ranked_features, desc="🔄 Processing features", unit="feat"), 1
    ):
        if f_i not in rankings.index:
            continue
        group = [f_i]
        group_name = "_".join(group)
        # group_perf is already calculated and stored in the rankings
        group_perf = rankings.loc[f_i, eval_criterion]
        rankings = rankings.drop(f_i)

        logger.info(
            f"\n🔍 {i}/{len(rankings)} - Starting new group {group_name} with [{f_i}] ({eval_criterion}: {group_perf:.4f})."
        )

        while len(group) < max_n:
            results = Parallel(n_jobs=-1)(
                delayed(evaluate_candidate)(
                    f_j,
                    group,
                    group_perf,
                    group_name,
                    alg,
                    learn_set,
                    val_set,
                    eval_criterion,
                    label,
                    logger,
                )
                for f_j in rankings.index
            )
            results = [r for r in results if r is not None]
            if results:
                best_result = max(results, key=lambda x: x["new_group_perf"])
                if best_result["new_group_perf"] > group_perf:
                    # if best_result["new_group_perf"] > (group_perf * 1.02):
                    group = best_result["new_group"]
                    group_perf = best_result["new_group_perf"]
                    rankings = rankings.drop(best_result["f_j"])
                    logger.info(
                        f"✅ Added {best_result["f_j"]} to group → New group: {group}."
                    )
                else:
                    logger.info(f"🚫 No further improvement found for group: {group}.")
                    break
            else:
                logger.warning("No valid candidate results — stopping.")
                break

        if len(group) >= 2:
            X_of_N_list.append(group)
            logger.info(f"💾 Group saved: {group}")
        else:
            logger.info(f"⚠️ No valid group formed with {group}")
    if len(X_of_N_list) != 0:
        print_xofn_groups_stats(X_of_N_list)
    return X_of_N_list, get_avg_features(X_of_N_list)


def print_xofn_groups_stats(X_of_N_list: list[list[str]]) -> None:
    # print(f"XofN_groups: {X_of_N_list}")
    lengths = [len(sublist) for sublist in X_of_N_list]
    length_counts = Counter(lengths)
    for length, count in sorted(length_counts.items()):
        print(f"XofN groups with {length} features: {count}")


def get_avg_features(X_of_N_list: list[list[str]]) -> float:
    if len(X_of_N_list) == 0:
        return 0
    total_items = sum(len(sublist) for sublist in X_of_N_list)
    total_lists = len(X_of_N_list)
    avg_features = total_items / total_lists
    return avg_features


def evaluate_candidate(
        f_j,
        group: list,
        group_perf,
        group_name: str,
        alg: Literal["DT", "RF"],
        learn_set: pd.DataFrame,
        val_set: pd.DataFrame,
        eval_criterion: str,
        label: str,
        logger: logging.Logger,
):
    try:
        new_group = group + [f_j]
        new_group_name = "_".join(new_group)
        logger.info(f"➡️  Trying {f_j} with current group {group}.")
        logger.info(f"Calculating {group_name} vs {new_group_name} performance.")
        new_group_perf = get_new_group_perf(
            alg,
            new_group,
            new_group_name,
            learn_set,
            val_set,
            eval_criterion,
            label,
        )
        logger.info(
            f"Comparing {group_name}:{group_perf:.4f} vs {new_group_name}:{new_group_perf:.4f}."
        )
        return {"f_j": f_j, "new_group_perf": new_group_perf, "new_group": new_group}
    except Exception as e:
        logger.error(f"Error with f_j={f_j}: {e}")
        return None


def get_new_group_perf(
        alg: Literal["DT", "RF"],
        new_group: list[str],
        new_group_name: str,
        learn_set: pd.DataFrame,
        val_set: pd.DataFrame,
        eval_criterion: str,
        label: str,
):
    new_group_learn_set = learn_set[new_group + [label]]
    new_group_learn_set = group_features(
        new_group_learn_set, [label], [new_group], False
    )
    new_group_val_set = val_set[new_group + [label]]
    new_group_val_set = group_features(new_group_val_set, [label], [new_group], False)
    model = train(alg, new_group_learn_set, [label], False)
    # TODO: add additional scores here e.g. mi_scores = mutual_info_classif(X[[feature1, feature2]], y)
    performance_df = test(model, new_group_val_set, [label], False)
    performance_df = performance_df.reset_index(drop=True)
    new_group_perf = performance_df.iloc[0].to_dict()
    if DEBUG:
        print(f"{new_group_name}: {new_group_perf[eval_criterion]}")
    return new_group_perf[eval_criterion]


def get_filtered_results(
        results: pd.DataFrame,
        label: str,
        ranking_criterion: str,
        include_original_features: bool,
):
    include = "with_org" if include_original_features else "no_org"
    filtered_df = results[
        (results["index"] == label)
        & (results["ranking_criterion"] == ranking_criterion)
        & (results["include_original_features"] == include)
        ]
    return filtered_df


def get_cv_mean_results(
        results: pd.DataFrame,
        labels: list[str],
        ranking_criteria: list[str],
        include_original_features_options: list[bool],
):
    cv_mean_results = pd.DataFrame([])
    for label in labels:
        for ranking_criterion in ranking_criteria:
            for include in include_original_features_options:
                filtered_results = get_filtered_results(
                    results, label, ranking_criterion, include
                )
                mean_row = {
                    "index": filtered_results["index"].iloc[0],
                    "ranking_criterion": filtered_results["ranking_criterion"].iloc[0],
                    "include_original_features": filtered_results[
                        "include_original_features"
                    ].iloc[0],
                    "roc_auc": filtered_results["roc_auc"].mean(),
                    "training_algorithm": filtered_results["training_algorithm"].iloc[
                        0
                    ],
                    "eval_criterion": filtered_results["eval_criterion"].iloc[0],
                    "max_size": filtered_results["max_size"].iloc[0],
                }
                cv_mean_results = pd.concat([cv_mean_results, pd.DataFrame([mean_row])])

    pivoted_cv_mean_results = cv_mean_results.pivot(
        index=[
            "index",
            "ranking_criterion",
            "training_algorithm",
            "eval_criterion",
            "max_size",
        ],
        columns="include_original_features",
        values="roc_auc",
    ).reset_index()
    pivoted_cv_mean_results.columns.name = None
    return pivoted_cv_mean_results


def get_baseline_results(all_labels_cv_results: pd.DataFrame, labels: list[str], labels_to_names: dict) -> pd.DataFrame:
    mean_results = {}
    for label in labels:
        label_results = all_labels_cv_results[label]
        mean_row = {
            "index": label,
            "roc_auc": label_results["roc_auc"].mean(),
        }
        mean_results[label] = mean_row["roc_auc"]

    rounded_results = {label: round(score, 4) for label, score in mean_results.items()}
    rounded_results["Macro AUROC"] = round(sum(mean_results.values()) / len(mean_results), 4)
    mean_results_df = pd.DataFrame.from_dict(rounded_results, orient='index', columns=['cv_auroc'])
    for label in rounded_results:
        if label in labels_to_names:
            print(f"{labels_to_names[label]} [{label}]: {round(rounded_results[label], 4)}")
        else:
            print(f"{label}: {round(rounded_results[label], 4)}")
    mean_results_df.index.name = 'ADR'
    return mean_results_df


def drop_column_importance(feature, X, y, rf, cv, baseline_acc, scoring):
    # print(f"dropping {feature}")
    X_drop = X.drop(columns=[feature])
    scores = cross_val_score(rf, X_drop, y, cv=cv, scoring=scoring, n_jobs=-1)
    # print(f"{feature} score: {np.mean(scores)}:")
    return baseline_acc - np.mean(scores)


def get_feature_importance(full_dataset, current_label, scoring, k):
    features = [col for col in full_dataset if col != current_label]
    print(len(features))
    print(current_label in features)

    X = full_dataset.drop(columns=[current_label])
    y = full_dataset[current_label]

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    # Baseline accuracy (all features)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    baseline_scores = cross_val_score(rf, X, y, cv=cv, scoring=scoring)
    baseline_acc = np.mean(baseline_scores)
    print(f"baseline AUROC: {baseline_acc}")

    # Drop-column importance
    drop_importances = Parallel(n_jobs=-1)(
        delayed(drop_column_importance)(feature, X, y, rf, cv, baseline_acc, scoring)
        for feature in tqdm(features, desc="Drop-column importance")
    )

    # Results
    importance_df = pd.DataFrame({
        'feature': features,
        'drop_importance': drop_importances
    }).sort_values(by='drop_importance', ascending=False)
    return importance_df


def get_top_k_features(full_df: pd.DataFrame, importance_df: pd.DataFrame, percent=0.6):
    n_keep = int(len(importance_df) * percent)
    print(f"Selecting {percent * 100}% of the features: {n_keep}")
    keep = importance_df["feature"].head(n_keep).tolist()
    feature_columns = [f for f in full_df.columns if f in keep]
    xofn_features_selected = [f for f in feature_columns if f.count("_") >= 2]
    print(f"{len(xofn_features_selected)} XofN features selected")
    return feature_columns


def generate_XofN_list_multi(
        training_set: pd.DataFrame,
        rankings: pd.DataFrame,
        eval_criteria: list[str],
        max_n: int,
        labels: list[str],
        pruning: bool,
        logger: logging.Logger,
        clus_path: str,
        tmp: str
):
    gen_XofN_start = time.perf_counter()
    print(
        f"\ngenerate_XofN_list -> Generating groupings based on eval_criterion:{eval_criteria}."
    )
    X_of_N_list = []
    ranked_features = list(rankings.index)  # Ensure consistent indexing
    for i, f_i in enumerate(
            tqdm(ranked_features, desc="🔄 Processing features", unit="feat"), 1
    ):
        if f_i not in rankings.index:
            continue
        group = [f_i]
        group_name = "_".join(group)
        group_path = f"{tmp}/{group_name}.csv"
        training_set[group + labels].to_csv(group_path, index=False)
        original_res, pruned_res, training_time = run_PCT(clus_path, group_path, labels, eval_criteria,
                                                          test_dataset_split=0.2)
        if pruning:
            group_perf = pruned_res
        else:
            group_perf = original_res
        group_perf = get_PCT_results(group_perf, eval_criteria)
        os.remove(group_path)
        rankings = rankings.drop(f_i)
        logger.info(
            f"\n🔍 {i}/{len(rankings)} - Starting new group {group_name} with [{f_i}] ({eval_criteria}: {group_perf})."
        )

        while len(group) < max_n:
            results = Parallel(n_jobs=-1)(
                delayed(evaluate_candidate_multi)(
                    clus_path,
                    f_j,
                    group,
                    group_name,
                    training_set,
                    eval_criteria,
                    labels,
                    pruning,
                    logger,
                    tmp
                )
                for f_j in rankings.index
            )
            results = [r for r in results if r is not None]
            shutil.rmtree(tmp)
            os.makedirs(tmp, exist_ok=True)
            remove_PCT_files("XofN_wrapper/tmp")
            if results:
                results_df = pd.DataFrame(results)
                fj_rankings = get_fj_rankings(results_df)
                best_result = fj_rankings.loc[fj_rankings["Avg_Rank"].idxmin()]
                best_feature_name = best_result["Feature"]
                best_feature_results = results_df[results_df["Feature"] == best_feature_name]
                best_feature_results_dict = best_feature_results.iloc[0].to_dict()
                if is_better(best_feature_results_dict, group_perf, eval_criteria) == 1:
                    group = best_result["new_group"]
                    group_perf = best_feature_results_dict
                    rankings = rankings.drop(best_feature_results_dict["Feature"])
                    logger.info(
                        f"✅ Added {best_feature_results_dict["Feature"]} to group → New group: {group}."
                    )
                else:
                    logger.info(f"🚫 No further improvement found for group: {group}.")
                    break
            else:
                logger.warning("No valid candidate results — stopping.")
                break

        if len(group) >= 2:
            X_of_N_list.append(group)
            logger.info(f"💾 Group saved: {group}")
        else:
            logger.info(f"⚠️ No valid group formed with {group}")
    if len(X_of_N_list) != 0:
        print_xofn_groups_stats(X_of_N_list)
    gen_XofN_end = time.perf_counter()
    return X_of_N_list, get_avg_features(X_of_N_list), gen_XofN_end - gen_XofN_start


def get_fj_rankings(results_df: pd.DataFrame):
    ranking = pd.DataFrame()
    ranking["Feature"] = results_df["Feature"]
    ranking["new_group"] = results_df["new_group"]

    # HammingLoss → lower is better
    ranking["Rank_HammingLoss"] = results_df["HammingLoss"].rank(method="min", ascending=True)

    # SubsetAccuracy → higher is better
    ranking["Rank_SubsetAccuracy"] = results_df["SubsetAccuracy"].rank(method="min", ascending=False)

    # averageAUROC → higher is better
    ranking["Rank_AUROC"] = results_df["averageAUROC"].rank(method="min", ascending=False)

    ranking["Avg_Rank"] = ranking[["Rank_HammingLoss", "Rank_SubsetAccuracy", "Rank_AUROC"]].mean(axis=1)
    ranking_sorted = ranking.sort_values(by="Avg_Rank", ascending=True).reset_index(drop=True)
    return ranking_sorted


def evaluate_candidate_multi(
        clus_path: str,
        f_j,
        group: list,
        group_name: str,
        training_set: pd.DataFrame,
        eval_criteria: list[str],
        labels: list[str],
        pruning: bool,
        logger: logging.Logger,
        tmp: str
):
    try:
        new_group = group + [f_j]
        new_group_name = "_".join(new_group)
        logger.info(f"➡️  Trying {f_j} with current group {group}.")
        logger.info(f"Calculating {group_name} vs {new_group_name} performance.")
        new_group_perf = get_new_group_perf_multi(
            clus_path,
            new_group,
            new_group_name,
            training_set,
            eval_criteria,
            labels,
            pruning,
            tmp
        )
        new_group_perf = get_PCT_results(new_group_perf, eval_criteria)
        logger.info(
            f"F_j:{f_j} - {new_group_name}:{new_group_perf}."
        )
        new_group_perf["Feature"] = f_j
        new_group_perf["new_group"] = new_group
        return new_group_perf
    except Exception as e:
        logger.error(f"Error with f_j={f_j}: {e}")
        return None


def get_new_group_perf_multi(
        clus_path: str,
        new_group: list[str],
        new_group_name: str,
        training_set: pd.DataFrame,
        eval_criteria: list[str],
        labels: list[str],
        pruning: bool,
        tmp: str
):
    new_group_learn_set = merge_features_sum(
        training_set[new_group + labels], new_group, labels
    )
    new_group_path = f"{tmp}/{new_group_name}.csv"
    new_group_learn_set.to_csv(new_group_path, index=False)
    original_res, pruned_res, training_time = run_PCT(clus_path, new_group_path, labels, eval_criteria,
                                                      test_dataset_split=0.2)
    if pruning:
        new_group_perf = pruned_res
    else:
        new_group_perf = original_res
    if DEBUG:
        print(f"{new_group_name}: {new_group_perf}")
    return new_group_perf


def is_better(feature1, feature2, measures):
    score = 0

    for m in measures:
        f1_val = feature1[m]
        f2_val = feature2[m]

        # Compare correctly depending on whether higher is better
        if m == "SubsetAccuracy" or m == "averageAUROC":
            if f1_val > f2_val:
                score += 1
            elif f1_val < f2_val:
                score -= 1
        else:  # lower is better
            if f1_val < f2_val:
                score += 1
            elif f1_val > f2_val:
                score -= 1

        # Decide majority vote
    if score > 0:
        return 1  # feature1 wins
    elif score < 0:
        return -1  # feature2 wins
    else:
        return 0  # tie


def get_PCT_results(perf_df: pd.DataFrame, eval_criteria: list[str]):
    perf_df = perf_df.T
    perf = {}
    if "nodes" in perf_df.columns:
        if isinstance(perf_df["nodes"], pd.Series):
            perf["nodes"] = perf_df["nodes"].values[0]
        else:
            perf["nodes"] = perf_df["nodes"]
    if "leaves" in perf_df.columns:
        if isinstance(perf_df["leaves"], pd.Series):
            perf["leaves"] = perf_df["leaves"].values[0]
        else:
            perf["leaves"] = perf_df["leaves"]
    for m in eval_criteria:
        if isinstance(perf_df[m], pd.Series):
            perf[m] = perf_df[m].values[0]
        else:
            perf[m] = perf_df[m]
    return perf


def generate_XofN_list_baseline(
        rankings: pd.DataFrame,
        max_n: int,
):
    gen_XofN_start = time.perf_counter()
    indices = list(rankings.index)
    XofN_groups = [indices[i:i + max_n] for i in range(0, len(indices), max_n)]
    gen_XofN_end = time.perf_counter()
    return XofN_groups, gen_XofN_end - gen_XofN_start


def generate_XofN_list_multi_custom(
        training_set: pd.DataFrame,
        rankings: pd.DataFrame,
        max_n: int,
        labels: list[str],
        logger: logging.Logger,
):
    gen_XofN_start = time.perf_counter()
    print(
        f"\ngenerate_XofN_list -> Generating groupings based on variance reduction:variance reduction."
    )
    X_of_N_list = []
    ranked_features = list(rankings.index)  # Ensure consistent indexing
    for i, f_i in enumerate(
            tqdm(ranked_features, desc="🔄 Processing features", unit="feat"), 1
    ):
        if f_i not in rankings.index:
            continue
        group = [f_i]
        group_name = "_".join(group)
        score, thr = best_variance_reduction_for_feature(training_set[[f_i]], training_set[labels],
                                                         min_samples_leaf=2)
        group_eval_score = {"best_variance_reduction": score}
        rankings = rankings.drop(f_i)
        print_message = f"\n🔍 {i}/{len(rankings)} - Starting new group {group_name} with [{f_i}] (variance reduction: {group_eval_score["best_variance_reduction"]})."
        logger.info(print_message)

        while len(group) < max_n:
            print_message = f"Comparing group eval: {group}: {group_eval_score["best_variance_reduction"]} with fjs"
            logger.info(print_message)
            results = Parallel(n_jobs=-1)(
                delayed(evaluate_candidate_multi_custom)(
                    f_j,
                    group,
                    group_name,
                    training_set,
                    labels,
                    logger
                )
                for f_j in rankings.index
            )
            results = [r for r in results if r is not None]
            if results:
                best_result = max(results, key=lambda x: x["best_variance_reduction"])
                print_message = f" Comparing group eval: {group}: {group_eval_score["best_variance_reduction"]} against {best_result["new_group"]}: {best_result["best_variance_reduction"]}."
                logger.info(print_message)
                if best_result["best_variance_reduction"] > group_eval_score["best_variance_reduction"]:
                    group = best_result["new_group"]
                    group_eval_score = best_result
                    rankings = rankings.drop(best_result["Feature"])
                    print_message = f"✅ Added {best_result["Feature"]} to group → New group: {group}."
                    logger.info(print_message)
                else:
                    print_message = f"🚫 No further improvement found for group: {group}."
                    logger.info(print_message)
                    break
            else:
                print_message = "No valid candidate results — stopping."
                logger.info(print_message)
                break

        if len(group) >= 2:
            X_of_N_list.append(group)
            print_message = f"💾 Group saved: {group}"
            logger.info(print_message)
        else:
            print_message = f"⚠️ No valid group formed with {group}"
            logger.info(print_message)
    if len(X_of_N_list) != 0:
        print_xofn_groups_stats(X_of_N_list)
    gen_XofN_end = time.perf_counter()
    return X_of_N_list, get_avg_features(X_of_N_list), gen_XofN_end - gen_XofN_start


def evaluate_candidate_multi_custom(
        f_j,
        group: list,
        group_name: str,
        training_set: pd.DataFrame,
        labels: list[str],
        logger: logging.Logger,
):
    try:
        new_group = group + [f_j]
        new_group_name = "_".join(new_group)
        print_message = f"➡️  Trying {f_j} with current group {group}."
        logger.info(print_message)
        print_message = f"Calculating {group_name} vs {new_group_name} performance."
        logger.info(print_message)
        new_group_training_set = training_set[new_group + labels]
        new_group_training_set = merge_features_sum(
            new_group_training_set, new_group, labels
        )
        score, thr = best_variance_reduction_for_feature(new_group_training_set[[new_group_name]],
                                                         new_group_training_set[labels], min_samples_leaf=2)
        new_group_perf = {"best_variance_reduction": score}
        print_message = f"F_j:{f_j} - {new_group_name}:{new_group_perf["best_variance_reduction"]}."
        logger.info(print_message)
        new_group_perf["Feature"] = f_j
        new_group_perf["new_group"] = new_group
        return new_group_perf
    except Exception as e:
        print_message = f"Error with f_j={f_j}: {e}"
        logger.info(print_message)
        return None


import pandas as pd


def merge_features_sum(dataset: pd.DataFrame, feature_cols: list[str], label_cols: list[str]) -> pd.DataFrame:
    # Create a name for the merged column
    f_group_name = "_".join(feature_cols)

    # Compute the row-wise sum for the given features
    merged = dataset[feature_cols].sum(axis=1)

    # Combine with label columns
    result = pd.concat([merged.rename(f_group_name), dataset[label_cols]], axis=1)

    return result


def get_fold_results(res, eval_criteria, pruning, fold, include_original_features, XofN_groupings,
                     gen_XofN_time,
                     training_time, dataset_name):
    performance = get_PCT_results(res, eval_criteria)
    performance["pruning"] = pruning
    performance["fold"] = fold
    performance["include_original_features"] = (
        "with_org" if include_original_features else "no_org"
    )
    performance["avg_group_features"] = get_avg_features(XofN_groupings)
    performance["groups"] = len(XofN_groupings)
    performance["gen_XofN_time"] = gen_XofN_time
    performance["training_time"] = training_time
    performance["dataset"] = dataset_name

    printable_props = ['pruning', 'include_original_features'] + eval_criteria
    message = ""
    for prop in printable_props:
        message += f"{prop}: {performance[prop]}, "
    print(message)
    return performance


def get_table_results(rounded_final_grouped_res, measures, paths):
    rounded_final_grouped_res = rounded_final_grouped_res[rounded_final_grouped_res["pruning"] == True].copy()
    for measure in measures:
        rounded_final_grouped_res[measure] = rounded_final_grouped_res[measure].round(3)

    rounded_final_grouped_res["Nodes; Leaves"] = rounded_final_grouped_res["nodes"].round(1).astype(str) + "; " + \
                                                 rounded_final_grouped_res["leaves"].round(1).astype(str)

    rounded_final_grouped_res["#XofN; #Feat/XofN"] = rounded_final_grouped_res["groups"].round(1).astype(str) + "; " + \
                                                     rounded_final_grouped_res["avg_group_features"].round(1).astype(
                                                         str)
    if "#unused" in rounded_final_grouped_res.columns:
        rounded_final_grouped_res["# Ung. Feats"] = rounded_final_grouped_res["#unused"]
    else:
        rounded_final_grouped_res["# Ung. Feats"] = rounded_final_grouped_res["dataset"].apply(
            lambda d: [p["features"] for p in paths if p["dataset_name"] == d][0]
        ) - (rounded_final_grouped_res["groups"] * rounded_final_grouped_res["avg_group_features"])

    rounded_final_grouped_res["XofN time (s); PCT tr. time (s)"] = rounded_final_grouped_res["gen_XofN_time"].round(
        1).astype(
        str) + "; " + rounded_final_grouped_res["training_time"].round(1).astype(str)

    rounded_final_grouped_res = rounded_final_grouped_res.drop(
        columns=["pruning", "nodes", "leaves", "groups", "avg_group_features", "gen_XofN_time", "training_time"])
    rounded_final_grouped_res.sort_values(by=['dataset', 'include_original_features'], ascending=[False, True],
                                          inplace=True)
    return rounded_final_grouped_res


def generate_XofN_list_multi_jaccard(
        training_set: pd.DataFrame,
        rankings: pd.DataFrame,
        max_n: int,
        logger: logging.Logger,
        minimise: bool = False
):
    gen_XofN_start = time.perf_counter()
    print(
        f"\ngenerate_XofN_list -> Generating groupings based on variance reduction:variance reduction."
    )
    X_of_N_list = []
    ranked_features = list(rankings.index)  # Ensure consistent indexing
    for i, f_i in enumerate(
            tqdm(ranked_features, desc="🔄 Processing features", unit="feat"), 1
    ):
        if f_i not in rankings.index:
            continue
        group = [f_i]
        group_name = "_".join(group)
        rankings = rankings.drop(f_i)
        print_message = f"\n🔍 {i}/{len(rankings)} - Starting new group {group_name} with [{f_i}]."
        logger.info(print_message)

        while len(group) < max_n:
            print_message = f"Comparing Jaccard similarity: {group} with fjs"
            logger.info(print_message)
            results = Parallel(n_jobs=-1)(
                delayed(evaluate_candidate_multi_jaccard)(
                    f_j,
                    group,
                    group_name,
                    training_set,
                    logger
                )
                for f_j in rankings.index
            )
            results = [r for r in results if r is not None]
            if results:
                best_result = min(results, key=lambda x: x["jac_simil"]) if minimise else max(results, key=lambda x: x[
                    "jac_simil"])
                group = best_result["new_group"]
                rankings = rankings.drop(best_result["Feature"])
                print_message = f"✅ Added {best_result["Feature"]} to group → New group: {group}."
                logger.info(print_message)
            else:
                print_message = "No valid candidate results — stopping."
                logger.info(print_message)
                break
        if len(group) >= 2:
            X_of_N_list.append(group)
            print_message = f"💾 Group saved: {group}"
            logger.info(print_message)
        else:
            print_message = f"⚠️ No valid group formed with {group}"
            logger.info(print_message)
    if len(X_of_N_list) != 0:
        print_xofn_groups_stats(X_of_N_list)
    gen_XofN_end = time.perf_counter()
    return X_of_N_list, get_avg_features(X_of_N_list), gen_XofN_end - gen_XofN_start


import numpy as np
from scipy.spatial.distance import jaccard


def evaluate_candidate_multi_jaccard(
        f_j,
        group: list,
        group_name: str,
        training_set: pd.DataFrame,
        logger: logging.Logger,
):
    try:
        jac_simil = {}
        jac_simil["Fj"] = 0
        for f_k in group:
            jac_simil["Fj"] = jac_simil["Fj"] + eval_jac_simil(training_set[f_j], training_set[f_k])
        jac_simil["Fj"] = jac_simil["Fj"] / len(group)
        new_group = group + [f_j]
        new_group_name = "_".join(new_group)
        print_message = f"➡️  Trying {f_j} with current group {group}."
        logger.info(print_message)
        print_message = f"Calculating {group_name} vs {new_group_name} performance."
        logger.info(print_message)
        new_group_perf = {"jac_simil": jac_simil["Fj"]}
        print_message = f"F_j:{f_j} - {new_group_name}:{new_group_perf["jac_simil"]}."
        logger.info(print_message)
        new_group_perf["Feature"] = f_j
        new_group_perf["new_group"] = new_group
        return new_group_perf
    except Exception as e:
        print_message = f"Error with f_j={f_j}: {e}"
        logger.info(print_message)
        return None


def eval_jac_simil(f_j: pd.DataFrame, f_k: pd.DataFrame) -> float:
    a = f_j.to_numpy().astype(bool)
    b = f_k.to_numpy().astype(bool)
    return 1.0 - jaccard(a, b) # jaccard() calculates dissimilarity


def generate_XofN_list_filter_jaccard(
        training_set: pd.DataFrame,
        rankings: pd.DataFrame,
        max_n: int,
        labels: list[str],
        logger: logging.Logger,
):
    gen_XofN_start = time.perf_counter()
    print(
        f"\ngenerate_XofN_list -> Generating groupings based on variance reduction:variance reduction."
    )
    X_of_N_list = []
    ranked_features = list(rankings.index)  # Ensure consistent indexing
    for i, f_i in enumerate(
            tqdm(ranked_features, desc="🔄 Processing features", unit="feat"), 1
    ):
        if f_i not in rankings.index:
            continue
        group = [f_i]
        group_name = "_".join(group)
        rankings = rankings.drop(f_i)
        print_message = f"\n🔍 {i}/{len(rankings)} - Starting new group {group_name} with [{f_i}])."
        logger.info(print_message)

        while len(group) < max_n:
            print_message = f"Calculating Variance reduction and Jaccard scores"
            logger.info(print_message)
            vr_scores = Parallel(n_jobs=-1)(
                delayed(evaluate_candidate_multi_custom)(
                    f_j,
                    group,
                    group_name,
                    training_set,
                    labels,
                    logger
                )
                for f_j in rankings.index
            )
            jac_scores = Parallel(n_jobs=-1)(
                delayed(evaluate_candidate_multi_jaccard)(
                    f_j,
                    group,
                    group_name,
                    training_set,
                    logger
                )
                for f_j in rankings.index
            )
            vr_scores = [r for r in vr_scores if r is not None]
            jac_scores = [r for r in jac_scores if r is not None]
            if vr_scores and jac_scores:
                vr_scores = normalise_results("best_variance_reduction", vr_scores)
                jac_scores = normalise_results("jac_simil", jac_scores)
                combined_scores = combine_scores(jac_scores, vr_scores)
                best_result = max(combined_scores, key=lambda x: x["combined_score"])
                group = best_result["new_group"]
                rankings = rankings.drop(best_result["Feature"])
                print_message = f"✅ Added {best_result["Feature"]} to group → New group: {group}."
                logger.info(print_message)
            else:
                print_message = "No valid candidate results — stopping."
                logger.info(print_message)
                break

        if len(group) >= 2:
            X_of_N_list.append(group)
            print_message = f"💾 Group saved: {group}"
            logger.info(print_message)
        else:
            print_message = f"⚠️ No valid group formed with {group}"
            logger.info(print_message)
    if len(X_of_N_list) != 0:
        print_xofn_groups_stats(X_of_N_list)
    gen_XofN_end = time.perf_counter()
    return X_of_N_list, get_avg_features(X_of_N_list), gen_XofN_end - gen_XofN_start


def normalise_results(score_property: str, results: list):
    values = np.array([res[score_property] for res in results]).reshape(-1, 1)
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values).flatten()

    for res, v_scaled in zip(results, values_scaled):
        res[score_property] = v_scaled
    return results


def combine_scores(jac_data, var_data):
    combined_scores = []
    for jac, var in zip(jac_data, var_data):
        assert jac['Feature'] == var['Feature'], "Features do not match!"
        combined_score = (jac['jac_simil'] + var['best_variance_reduction']) / 2
        combined_scores.append({
            "Feature": jac['Feature'],
            "new_group": jac['new_group'],  # or var['new_group'], they are the same
            "combined_score": combined_score
        })
    return combined_scores
