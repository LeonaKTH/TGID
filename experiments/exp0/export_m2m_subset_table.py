import argparse
import csv
import json
from pathlib import Path


SUBSETS = [
    ("m2m_all", "M2M-All"),
    ("non_m2m", "Non-M2M"),
    ("cross_type", "C-Type"),
    ("same_type_multi_instance", "S-Type"),
    ("cross_role", "C-Role"),
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_nested(data, keys):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def score_for_run(result_root, run_name):
    result_path = result_root / run_name / "best_result.json"
    if not result_path.exists():
        return {label: None for _, label in SUBSETS}
    data = read_json(result_path)
    row = {}
    for key, label in SUBSETS:
        value = get_nested(data, ["test", "event", "m2m", key, "micro_f1"])
        row[label] = value * 100 if isinstance(value, (int, float)) else None
    return row


def format_value(value):
    return "--" if value is None else "{:.1f}".format(value)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Model", "Dataset"] + [label for _, label in SUBSETS]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_markdown(rows):
    fields = ["Model", "Dataset"] + [label for _, label in SUBSETS]
    print("| " + " | ".join(fields) + " |")
    print("|" + "|".join(["---"] * len(fields)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[field]) for field in fields) + " |")


def main():
    parser = argparse.ArgumentParser(description="Export M2M subset F1 table from best_result.json files.")
    parser.add_argument("--result-root", default="Result")
    parser.add_argument("--output", default="Result/m2m_subset_table.csv")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    runs = [
        ("EPAL", "ChFinAnn", "epal_chfinann"),
        ("TGID-Softmax-T", "ChFinAnn", "tgid_chfinann_softmax_t"),
        ("TGID-Softmax-K", "ChFinAnn", "tgid_chfinann_softmax_k"),
        ("TGID", "ChFinAnn", "tgid_chfinann_sigmoid"),
        ("EPAL", "DuEE-Fin", "epal_dueefin"),
        ("TGID-Softmax-T", "DuEE-Fin", "tgid_dueefin_softmax_t"),
        ("TGID-Softmax-K", "DuEE-Fin", "tgid_dueefin_softmax_k"),
        ("TGID", "DuEE-Fin", "tgid_dueefin_sigmoid"),
    ]

    rows = []
    for model, dataset, run_name in runs:
        scores = score_for_run(result_root, run_name)
        row = {"Model": model, "Dataset": dataset}
        row.update({label: format_value(scores[label]) for _, label in SUBSETS})
        rows.append(row)

    write_csv(Path(args.output), rows)
    print_markdown(rows)
    print("\nWrote {}".format(args.output))


if __name__ == "__main__":
    main()
