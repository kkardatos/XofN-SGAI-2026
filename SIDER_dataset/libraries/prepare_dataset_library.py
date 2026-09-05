import pandas as pd
from datetime import datetime
from typing import Literal
from SIDER_dataset.libraries.utils import get_project_path


def print_dataset_info(full_dataset: pd.DataFrame, labels: list[str]):
    dataset_info = {}
    dataset_info["Number of instances"] = len(full_dataset)
    if "CID" in full_dataset.columns:
        dataset_info["Number of drugs"] = full_dataset["CID"].unique().size
    dataset_info["Number of CPI features"] = len(get_cpis(full_dataset))
    dataset_info["Number of fingerprint features"] = len(get_fs(full_dataset))
    dataset_info["Number of features"] = len(get_features(full_dataset, labels))
    dataset_info["Number of ADRs"] = len(get_ADRs(full_dataset))
    for key, value in dataset_info.items():
        print(key + ":", value)
    return dataset_info


def get_cpis(full_dataset: pd.DataFrame):
    return [col for col in full_dataset.columns if col.startswith("cpi_")]


def get_fs(full_dataset: pd.DataFrame):
    return [col for col in full_dataset.columns if col.startswith("f_")]


def get_ADRs(full_dataset: pd.DataFrame):
    return [col for col in full_dataset.columns if col.startswith("se_")]


def get_features(full_dataset: pd.DataFrame, labels: list[str]):
    return [col for col in full_dataset.columns if col not in labels]


def clean_dataset(
        full_dataset: pd.DataFrame,
        cpi_threshold: int,
        fingerprint_threshold: int,
        labels: list[str],
):
    # Remove uninformative cpi features
    # Drop CPI columns where the number of instances with a value of 500 and above is less than 10
    CPI_columns = get_cpis(full_dataset)
    columns_to_keep = [
        col for col in CPI_columns if (full_dataset[col] > 499).sum() >= cpi_threshold
    ]

    # Keep only the selected columns that meet the condition, and all other columns in the DataFrame
    full_dataset = full_dataset.drop(columns=set(CPI_columns) - set(columns_to_keep))
    print("\nAfter removing uninformative cpi features")
    print_dataset_info(full_dataset, labels)
    full_dataset.head()

    # Drop uninformative fingerprint features
    # Drop fingerprint columns where the number of instances with a value of 1 and above is less than 10
    fs = get_fs(full_dataset)
    ones_count = full_dataset[fs].sum()

    # Drop columns where count of 1s and above is less than 10
    full_dataset = full_dataset.drop(
        columns=ones_count[ones_count < fingerprint_threshold].index
    )
    print("\nAfter removing uninformative fingerprint features")
    print_dataset_info(full_dataset, labels)
    full_dataset.head()

    # Remove uninformative features
    # Drop columns where all values are the same
    print("\nAfter removing where all values are the same")
    full_dataset = full_dataset.loc[:, full_dataset.nunique() > 1]
    print_dataset_info(full_dataset, labels)

    # Drop duplicates - keep 1
    print("\nAfter removing duplicates - keeping one")
    full_dataset = full_dataset.drop_duplicates()
    print_dataset_info(full_dataset, labels)
    return full_dataset


def get_ADR_frequency(full_cleaned_dataset: pd.DataFrame):
    ADR_frequency_dict = {}
    for ADR in get_ADRs(full_cleaned_dataset):
        ADR_frequency_dict[ADR] = full_cleaned_dataset[ADR].sum()
    ADR_frequency = pd.DataFrame(
        list(ADR_frequency_dict.items()), columns=["ADR", "count"]
    )
    ADR_frequency["ADR_code"] = ADR_frequency["ADR"].apply(
        lambda x: x.replace("se_", "")
    )
    print("ADRs:", ADR_frequency["ADR_code"].unique().size)
    ADR_frequency_frequency = ADR_frequency.sort_values(by="count", ascending=False)
    print(ADR_frequency_frequency.head())

    meddra = pd.read_csv("../datasets/original_datasets/SIDER-4.1/meddra.tsv", sep="\t")
    print("UMLS_ids:", meddra["UMLS_id"].unique().size)
    print("total rows:", len(meddra))
    meddra = meddra[meddra["MedDRA_ADR_level"] == "PT"]
    meddra = meddra.drop_duplicates()
    print("UMLS_ids:", meddra["UMLS_id"].unique().size)
    print("total rows:", len(meddra))
    print(meddra.head())

    ADR_frequency_with_names = pd.merge(
        ADR_frequency_frequency, meddra, left_on="ADR_code", right_on="UMLS_id"
    )
    ADR_frequency_with_names = ADR_frequency_with_names.drop(
        columns=["UMLS_id", "MedDRA_ADR_level", "MedDRA_ADR_id"]
    )
    print("UMLS_ids:", ADR_frequency_with_names["ADR_code"].unique().size)
    print("total rows:", len(ADR_frequency_with_names))
    ADR_frequency_with_names = (
        ADR_frequency_with_names.groupby(["ADR", "count", "ADR_code"])
        .agg(names=("MedDRA_ADR_name", list))
        .reset_index()
    )
    print("UMLS_ids:", ADR_frequency_with_names["ADR_code"].unique().size)
    print("total rows:", len(ADR_frequency_with_names))
    ADR_frequency_with_names = ADR_frequency_with_names.sort_values(
        by="count", ascending=False
    )
    ADR_frequency_with_names["percentage"] = ADR_frequency_with_names["count"] / len(
        full_cleaned_dataset
    )
    return ADR_frequency_with_names


def keep_6_most_frequent_ADRs(full_cleaned_dataset: pd.DataFrame):
    ADR_frequency_with_names = get_ADR_frequency(full_cleaned_dataset)
    cols_to_keep = (
            ["CID"]
            + get_cpis(full_cleaned_dataset)
            + get_fs(full_cleaned_dataset)
            + ADR_frequency_with_names.head(6)["ADR"].unique().tolist()
    )
    cols_to_drop = [
        col for col in full_cleaned_dataset.columns if col not in cols_to_keep
    ]
    full_cleaned_dataset_with_6_labels = full_cleaned_dataset.drop(columns=cols_to_drop)
    print_dataset_info(full_cleaned_dataset_with_6_labels)
    return full_cleaned_dataset_with_6_labels


def remove_inconsistent_duplicates(
        df: pd.DataFrame, group_cols: list, check_cols: list, output=False
):
    print("\nremove_inconsistent_duplicates")
    original_num_of_rows = len(df)
    df = df.drop_duplicates()
    print(
        f"  Dropping {original_num_of_rows - len(df)} duplicate rows (keep first). Remaining rows: {len(df)}"
    )
    grouped = df.groupby(group_cols)
    valid_rows = []
    invalid_rows = []

    for name, group in grouped:
        if all(group[col].nunique() == 1 for col in check_cols):
            valid_rows.append(group)
        else:
            invalid_rows.append(group)
    filtered_df = pd.concat(valid_rows).reset_index(drop=True)

    if invalid_rows:
        deleted_df = pd.concat(invalid_rows).reset_index(drop=True)
        print(
            f"  ❌ Deleting {len(deleted_df)} inconsistent rows (same values for features but different for labels). Remaining: {len(filtered_df)}"
        )
        if output:
            print(deleted_df.to_string())
    return filtered_df


def save_df_to_csv(df: pd.DataFrame, path: str, filename: str):
    full_filename = filename + "_" + datetime.now().strftime("%d-%m-%Y") + ".csv"
    df.to_csv(path + full_filename)


def keep_single_se(dataset: pd.DataFrame, se_to_predict: str):
    ses = get_ADRs(dataset)
    print("total ses:", len(ses))
    ses_to_drop = [col for col in ses if col != se_to_predict]
    print("dropping:", len(ses_to_drop))
    dataset = dataset.drop(columns=ses_to_drop)
    return dataset


def get_dataset(
        dataset_name: Literal["fingerprint", "CPI", "CPI+fingerprint"],
        labels_to_keep: list[str],
):
    """Join dataset joined with de2022interpretable to get ADRs.
    Keep target labels.
    """
    dataset = pd.read_csv("datasets/original_datasets/de2022interpretable.csv")

    if dataset_name == "fingerprint" or dataset_name == "CPI+fingerprint":
        fingerprints_df = pd.read_csv("datasets/original_datasets/fingerprints.csv")
        dataset = pd.merge(
            left=dataset, right=fingerprints_df, left_on="CID", right_on="CID1"
        )
        dataset = dataset.drop(columns=["CID1"])
        if dataset_name == "fingerprint":
            cpis = get_cpis(dataset)
            dataset = dataset.drop(columns=cpis)

    if labels_to_keep:
        ADRs = get_ADRs(dataset)
        ADRs_to_drop = [ADR for ADR in ADRs if ADR not in labels_to_keep]
        fingerprint_dataset = dataset.drop(columns=ADRs_to_drop)

    print_dataset_info(fingerprint_dataset, labels_to_keep)
    return fingerprint_dataset


def drop_uninformative_features(
        df: pd.DataFrame,
        feature_threshold: int,
        label_columns: list[str],
        binary_columns: list[str] = [],
        prob_columns: list[str] = [],
):
    print("\ndrop_uninformative_features")
    if binary_columns:
        # Drop binary (fingerprint) columns where the number of instances with a value of 1 and above is less than {feature_threshold}
        ones_count = df[binary_columns].sum()
        df = df.drop(columns=ones_count[ones_count < feature_threshold].index)
        remaining_binary_features = [
            column for column in binary_columns if column in df.columns
        ]
        print_message = f" Removing {len(ones_count[ones_count < feature_threshold])} uninformative binary features. Remaining binary features: {len(remaining_binary_features)}"
        print(print_message)

    if prob_columns:
        # Drop probability (CPI) columns where the number of instances with a value of 500 and above is less than {feature_threshold}
        columns_to_keep = [
            col for col in prob_columns if (df[col] > 499).sum() >= feature_threshold
        ]
        df = df.drop(columns=set(prob_columns) - set(columns_to_keep))
        remaining_probability_features = [
            column for column in prob_columns if column in df.columns
        ]
        print_message = f" Removing {len(ones_count[ones_count < feature_threshold])} uninformative probability scores features. Remaining probability features: {len(remaining_probability_features)}"
        print(print_message)

    # Drop columns where all values are the same
    original_col_count = df.shape[1]
    df = df.loc[:, df.nunique() > 1]
    removed_count = original_col_count - df.shape[1]
    remaining_features = [
        column for column in df.columns if column not in label_columns
    ]
    print_message = f" Removing {removed_count} uninformative features where all values are the same. Remaining features: {len(remaining_features)}"
    print(print_message)

    return (
        df,
        label_columns,
        remaining_features,
        locals().get("remaining_binary_features", []),
        locals().get("remaining_probability_features", []),
    )


def binarize_features(dataset: pd.DataFrame, features: list[str], threshold: int):
    binary_df = dataset.copy()
    binary_df[features] = (dataset[features] > threshold).astype(int)
    return binary_df


def get_feature_1_counts(dataset: pd.DataFrame, labels: list[str]):
    feature_df = dataset.drop(columns=labels)
    feature_frequency = (feature_df == 1).sum().sort_values(ascending=False)
    feature_frequency = feature_frequency.reset_index()
    feature_frequency.columns = ["feature", "count_ones"]
    return feature_frequency


def get_high_freq_labels():
    high_freq_labels_with_names = {
        "se_C0027497": "Nausea",  # 0.845
        "se_C0018681": "Headache",  # 0.783
        "se_C0011603": "Dermatitis",  # 0.767
        "se_C0015230": "Rash",  # 0.761
        "se_C0042963": "Vomiting",  # 0.760
        "se_C0012833": "Dizziness"  # 0.727
    }
    high_freq_labels = [ADR for ADR in high_freq_labels_with_names]
    return high_freq_labels_with_names, high_freq_labels


def get_mid_freq_labels():
    mid_freq_labels_with_names = {
        "se_C0020538": "Hypertension",  # 555 0.398
        "se_C0008031": "Chest pain",  # 540 0.387
        "se_C0003467": "Anxiety",  # 538 0.386
        "se_C0036572": "Convulsion",  # 509 0.365
        "se_C0030252": "Palpitations",  # 500 0.359
        "se_C0027769": "Nervousness",  # 431 0.309
    }
    mid_freq_labels = [ADR for ADR in mid_freq_labels_with_names]
    return mid_freq_labels_with_names, mid_freq_labels


def get_low_freq_labels():
    low_freq_labels_with_names = {
        "se_C0020580": "Hypoaesthesia",  # 354 0.254
        "se_C0267792": "Hepatobiliary disease",  # 339 0.243
        "se_C0018524": "Hallucination",  # 338 0.242
        "se_C0031117": "Neuropathy peripheral",  # 333 0.239
        "se_C0035078": "Renal failure",  # 324 0.232
        "se_C0011570": "Depression",  # 318 0.228
    }
    low_freq_labels = [ADR for ADR in low_freq_labels_with_names]
    return low_freq_labels_with_names, low_freq_labels


def get_cardiac_labels():
    cardiac_labels_with_names = {
        "se_C0016382": "Flushing,",  # 0.301	417
        "se_C0018799": "Cardiac disorder",  # 0.291  406
        "se_C0003811": "Arrhythmia",  # 0.29	402
        "se_C0428977": "Bradycardia",  # 0.256	355
        "se_C0027051": "Myocardial infarction",  # 0.236	328
        "se_C0018790": "Cardiac arrest",  # 0.166 	232
    }
    cardiac_labels = [ADR for ADR in cardiac_labels_with_names]
    return cardiac_labels_with_names, cardiac_labels


def get_ten_labels():
    ten_labels_with_names = {
        "se_C0011991": "Diarrhoea",  # 0.687    957
        "se_C0000737": "Abdominal pain",  # 0.580   808
        "se_C0687713": "Gastrointestinal pain",  # 0.548    728
        "se_C0030193": "Pain",  # 0.520     725
        "se_C0009806": "Constipation",  # 0.493     687
        "se_C0917801": "Insomnia",  # 0.475     662
        "se_C0020649": "Hypotension",  # 0.467   652
        "se_C0027765": "Nervous system disorder",  # 0.440  614
        "se_C0039231": "Tachycardia",  # 0.417  581
        "se_C0231528": "Myalgia",  # 0.399  556
    }
    ten_labels = [ADR for ADR in ten_labels_with_names]
    return ten_labels_with_names, ten_labels


def get_ten_labels_mid():
    ten_mid_labels_with_names = {
        "se_C0009676": "Confusional state",  # 0.385 537
        "se_C0041657": "Loss of consciousness",  # 0.374 521
        "se_C0002994": "Angioedema",  # 0.373    520
        "se_C0042571": "Vertigo",  # 0.357   498
        "se_C0004604": "Back pain",  # 0.351 489
        "se_C0041834": "Erythema",  # 0.341  475
        "se_C0085631": "Agitation",  # 0.335    467
        "se_C0040822": "Tremor",  # 0.327    456
        "se_C0042373": "Angiopathy",  # 0.322    449
        "se_C0021053": "Immune system disorder",  # 0.316    440
    }
    ten_mid_labels = [ADR for ADR in ten_mid_labels_with_names]
    return ten_mid_labels_with_names, ten_mid_labels


def get_labels_name(labels: list[str]) -> str:
    if "se_C0027497" in labels:
        return "high_freq"
    elif "se_C0027769" in labels:
        return "mid_freq"
    elif "se_C0020580" in labels:
        return "low_freq"
    elif "se_C0016382" in labels:
        return "cardiac"
    elif "se_C0011991" in labels:
        return "ten"
    elif "se_C0009676" in labels:
        return "ten_mid"
    return None


# to add datasets:
# add get_labels()
# add condition in get_labels_name
# add to get_dataset_paths
# create dataset in prepare_multilabel_datasets.ipynb
def get_dataset_paths(verbose: bool = False):
    high_freq_labels_with_names, high_freq_labels = get_high_freq_labels()
    mid_freq_labels_with_names, mid_freq_labels = get_mid_freq_labels()
    low_freq_labels_with_names, low_freq_labels = get_low_freq_labels()
    cardiac_labels_with_names, cardiac_labels = get_cardiac_labels()
    ten_labels_with_names, ten_labels = get_ten_labels()
    ten_mid_labels_with_names, ten_mid_labels = get_ten_labels_mid()
    labels = [high_freq_labels, mid_freq_labels, low_freq_labels, cardiac_labels, ten_labels, ten_mid_labels]
    dataset_names = ["fingerprint", "CPI", "CPI+fingerprint"]

    paths = []
    for label_set in labels:
        for dataset_name in dataset_names:
            if verbose:
                print(f"Processing {dataset_name} for {len(label_set)} labels: {label_set}")
            dataset_name_with_labels = f"{dataset_name}_{get_labels_name(label_set)}"
            dataset_path = f"{get_project_path()}/sep/SIDER_dataset/datasets/clean_multi_label_datasets/{dataset_name_with_labels}.csv"
            dataset = pd.read_csv(dataset_path)
            if verbose:
                print(dataset_name_with_labels)
                print(dataset_path, "\n")
            paths.append(
                {"dataset_path": dataset_path, "dataset_name": dataset_name_with_labels, "label_set": label_set,
                 "features": (len(dataset.columns) - len(label_set)),
                 "original_features": get_features(dataset, label_set)})
    paths.sort(key=lambda x: x["dataset_name"])
    return paths
