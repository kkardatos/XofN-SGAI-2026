from SIDER_dataset.libraries.prepare_dataset_library import get_features, get_ADRs, get_cpis, get_fs
import pandas as pd
import os
from pathlib import Path
import arff
import subprocess
import re
import numpy as np
import time


def run_PCT(clus_path: str, dataset_path: str, labels: list[str], measures: list[str], test_dataset_path: str = "",
            test_dataset_split: float = 0.0):
    training_start = time.perf_counter()

    arff_path = save_to_arff(dataset_path, labels)
    test_arff_path = ""
    if test_dataset_path:
        test_arff_path = save_to_arff(test_dataset_path, labels)
    settings_path = create_settings_file(dataset_path, labels, arff_path, test_arff_path, test_dataset_split)
    res_path = settings_path.replace(".s", ".xval")
    if test_dataset_path or test_dataset_split != 0.0:
        res_path = settings_path.replace(".s", ".out")

    command = [
        "java",
        "-jar",
        str(Path(clus_path)),
        "-xval",
        str(Path(settings_path)),
    ]
    if test_dataset_path or test_dataset_split != 0.0:
        command = [
            "java",
            "-jar",
            str(Path(clus_path)),
            str(Path(settings_path)),
        ]

    # print(command)
    process = subprocess.run(command, capture_output=True, text=True)

    # Print the output
    # print("STDOUT:", process.stdout)
    # print("STDERR:", process.stderr)

    original_res, pruned_res = get_measures_for_multi_label(res_path, measures)
    training_end = time.perf_counter()
    return original_res, pruned_res, training_end - training_start


def remove_PCT_files(dir: str) -> None:
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if file.endswith(".arff") or file.endswith(".out") or file.endswith(".s") or file.endswith(".xval"):
            os.remove(file_path)


def move_columns_to_end(df: pd.DataFrame, columns_to_move: list[str]) -> pd.DataFrame:
    columns = [
                  col for col in df.columns if col not in columns_to_move
              ] + columns_to_move
    return df[columns]


def get_column_positions(
        dataset: pd.DataFrame,
        columns_to_find: list[str],
):
    positions = [list(dataset.columns).index(col) + 1 for col in columns_to_find]
    if len(positions) == 1:
        return str(positions[0])

    smallest = min(positions)
    biggest = max(positions)
    return f"{smallest}-{biggest}"


def get_absolute_path(local_path):
    local_path = Path(local_path)
    absolute_path = local_path.resolve()
    return absolute_path


def create_settings_file(dataset_path: str, labels: list[str], arff_path: str, test_arff_path: str = "",
                         test_split: float = 0.0):
    dataset = pd.read_csv(dataset_path)
    test = "XVal = 10"
    if test_arff_path:
        test = f"TestSet = {test_arff_path}"
    if test_split != 0.0:
        test = f"TestSet = {test_split}"
    features = get_features(dataset, labels)
    descriptive = get_column_positions(dataset, features)
    target = get_column_positions(dataset, labels)
    settings_path = arff_path.replace(".arff", "_settings.s")
    content = f"""[General]
RandomSeed = 43

[Data]
File = {arff_path}
{test}

[Attributes]
Descriptive = {descriptive}
Target = {target}
Clustering = {target}

[Tree]
Heuristic = VarianceReduction
"""

    with open(settings_path, "w", encoding="utf-8") as file:
        file.write(content)

    return settings_path


def save_to_arff(csv_path: str, labels: list[str]):
    dataset = pd.read_csv(csv_path)
    dataset = move_columns_to_end(dataset, labels)
    arff_path = csv_path.replace(".csv", ".arff")
    arff.dump(
        arff_path,
        dataset.values,
        relation="_".join(labels),
        names=dataset.columns,
    )

    replace_dict = {
        "np.int64(": "",
        ")": "",
        "np.float64(1.0": ""
    }

    for label in labels:
        replace_dict[f"@attribute {label} integer"] = f"@attribute {label}" + " {0, 1}"

    for key, value in replace_dict.items():
        replace_in_file(arff_path, key, value)

    return arff_path


def replace_in_file(file_path, old_string, new_string):
    # print(f"replace '{old_string}' with '{new_string}'")
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    content = content.replace(old_string, new_string)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def get_measures_for_multi_label(file_path: str, measures: list[str]):
    num_next_lines = 3
    search_texts = measures
    original_results = {text: None for text in search_texts}  # Initialize dictionary
    pruned_results = {text: None for text in search_texts}  # Initialize dictionary

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()  # Read all lines into a list

    if "FOne" in measures or "Precision" in measures or "Recall" in search_texts:
        pruned_precision, pruned_recall, pruned_f1 = get_FOne_Precision_Recall(lines, "Pruned")
        original_precision, original_recall, original_f1 = get_FOne_Precision_Recall(lines, "Original")
        original_results["FOne"] = original_f1
        original_results["Precision"] = original_precision
        original_results["Recall"] = original_recall
        pruned_results["FOne"] = pruned_f1
        pruned_results["Precision"] = pruned_precision
        pruned_results["Recall"] = pruned_recall

    for search_text in search_texts:
        last_match = None

        for i, line in enumerate(lines):
            if search_text in line:
                # Capture the matching line + next 'num_next_lines' lines
                snippet = lines[i: i + num_next_lines + 1]
                last_match = snippet  # Store the last occurrence

        if last_match:
            # Extract only the last numeric value from 'Original:' and 'Pruned:'
            for line in last_match:
                if "Original" in line:
                    original_match = re.search(
                        r"Original\s*:\s*(?:\[.*\]:\s*)?(NaN|[\d.E+-]+)", line
                    )
                    if original_match:
                        value = original_match.group(1)
                        original_results[search_text] = 0.0 if value == "NaN" else float(value)
                elif "Pruned" in line:
                    pruned_match = re.search(
                        r"Pruned\s*:\s*(?:\[.*\]:\s*)?(NaN|[\d.E+-]+)", line
                    )
                    if pruned_match:
                        value = pruned_match.group(1)
                        pruned_results[search_text] = 0.0 if value == "NaN" else float(value)
    original_nodes, original_leaves = get_nodes_and_leaves(file_path, "Original")
    pruned_nodes, pruned_leaves = get_nodes_and_leaves(file_path, "Pruned")
    original_results["nodes"] = original_nodes
    original_results["leaves"] = original_leaves
    pruned_results["nodes"] = pruned_nodes
    pruned_results["leaves"] = pruned_leaves
    original_df = pd.DataFrame.from_dict(original_results, orient="index")
    pruned_df = pd.DataFrame.from_dict(pruned_results, orient="index")
    return original_df, pruned_df


def get_FOne_Precision_Recall(lines, model):
    in_section = False
    in_model = False
    matrix_rows = []

    for i, line in enumerate(lines):
        # Enter the right section
        if "Testing error" in line:
            in_section = True

        # Leave section if a new big header starts
        if in_section and line.strip().endswith("*************"):
            break

        # Enter the right model
        if in_section and line.strip().startswith(model + ":"):
            in_model = True

        # Grab matrix rows
        if in_model and "|" in line:
            stripped = line.strip()

            # Match rows like: 1 | 12 | 28 | 40
            parts = stripped.split("|")
            if len(parts) >= 4 and parts[0].strip() in ("0", "1"):
                real = parts[0].strip()
                pred1 = int(parts[1])
                pred0 = int(parts[2])
                matrix_rows.append((real, pred1, pred0))

                if len(matrix_rows) == 2:
                    break

    if len(matrix_rows) != 2:
        raise ValueError("Confusion matrix not found")

    # Unpack
    row_1 = next(r for r in matrix_rows if r[0] == "1")
    row_0 = next(r for r in matrix_rows if r[0] == "0")

    TP = row_1[1]
    FN = row_1[2]
    FP = row_0[1]
    TN = row_0[2]

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def get_nodes_and_leaves(file_path: str, originalOrPruned: str):
    pattern = re.compile(
        fr"\s*{originalOrPruned}:\s*Nodes\s*=\s*(\S+);\s*Leaves\s*=\s*(\S+)",
        re.IGNORECASE
    )

    nodes_values, leaves_values = [], []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    nodes_values.append(float(match.group(1).replace(";", "")))
                    leaves_values.append(float(match.group(2).replace(";", "")))
                except ValueError:
                    continue

    if not nodes_values or not leaves_values:
        raise ValueError("No nodes or leaves found")

    if len(nodes_values) > 1:
        return np.mean(nodes_values), np.mean(leaves_values)

    return nodes_values[0], leaves_values[0]
