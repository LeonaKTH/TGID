import argparse
from collections import defaultdict

from utils import extract_tuples, load_gold_categories, write_json


EMPTY = {"tp": 0, "fp": 0, "fn": 0}


def score(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser(description="Experiment 0 subset F1 by gold argument category.")
    parser.add_argument("--gold_data", required=True)
    parser.add_argument("--pred_data", required=True)
    parser.add_argument("--output", default="outputs/exp0/subset_f1.json")
    args = parser.parse_args()

    _, categories_by_doc = load_gold_categories(args.gold_data)
    gold_tuples, gold_tuple_to_arg_key = extract_tuples(args.gold_data)
    pred_tuples, _ = extract_tuples(args.pred_data)

    tuple_category = {}
    for tpl, arg_key in gold_tuple_to_arg_key.items():
        doc_id = tpl[0]
        tuple_category[tpl] = categories_by_doc.get(doc_id, {}).get(arg_key, {}).get("category", "SINGLE")

    counts = defaultdict(lambda: dict(EMPTY))
    for tpl in gold_tuples:
        category = tuple_category.get(tpl, "SINGLE")
        if tpl in pred_tuples:
            counts[category]["tp"] += 1
            counts["OVERALL"]["tp"] += 1
        else:
            counts[category]["fn"] += 1
            counts["OVERALL"]["fn"] += 1

    for tpl in pred_tuples:
        if tpl not in gold_tuples:
            # Experiment 0 first pass: an unmatched FP has no reliable gold
            # argument category, so it is counted only in OVERALL. Matched
            # predictions inherit the corresponding gold tuple category above.
            counts["OVERALL"]["fp"] += 1

    output = {}
    for category in ["SINGLE", "SAME_TYPE_MULTI", "CROSS_TYPE_M2M", "OVERALL"]:
        output[category] = score(counts[category])
    write_json(args.output, output)


if __name__ == "__main__":
    main()
