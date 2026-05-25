import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def tensorize(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float)


def stringify_key(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def continuous_gate(record, gate):
    if record.get("route_logits") is not None:
        logits = tensorize(record["route_logits"])
        if gate == "softmax":
            return F.softmax(logits, dim=-1)
        score = torch.sigmoid(logits)
        return score / score.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    prob = tensorize(record["route_prob"])
    return prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def pair_target_sensitivity(gate_vector, type_a, type_b):
    return float(1.0 - F.cosine_similarity(gate_vector[type_a], gate_vector[type_b], dim=0).item())


def pair_soft_overlap(gate_vector, type_a, type_b):
    return float(torch.minimum(gate_vector[type_a], gate_vector[type_b]).sum().item())


def pair_top1_collision(gate_vector, type_a, type_b):
    return float(torch.argmax(gate_vector[type_a]).item() == torch.argmax(gate_vector[type_b]).item())


def mean(values):
    return sum(values) / len(values) if values else None


def load_eval_dump(result_root, run_name):
    path = result_root / run_name / "best_eval_dump.pkl"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def gate_name(eval_dump):
    config = eval_dump.get("config", {})
    return str(config.get("tgid_gate", "sigmoid"))


def collect_pair_vectors(eval_dump):
    gate = gate_name(eval_dump)
    pair_vectors = {}
    for record in eval_dump.get("route_records", []):
        if not record.get("is_m2m_all", False):
            continue
        gold_type_ids = sorted({int(x) for x in record.get("gold_event_type_ids", [])})
        if len(gold_type_ids) < 2:
            continue
        gate_vector = continuous_gate(record, gate)
        doc_id = str(record.get("doc_id", ""))
        arg_key = stringify_key(record.get("arg_key", ""))
        for index, type_a in enumerate(gold_type_ids):
            for type_b in gold_type_ids[index + 1:]:
                if type_a >= gate_vector.shape[0] or type_b >= gate_vector.shape[0]:
                    continue
                pair_key = (doc_id, arg_key, int(type_a), int(type_b))
                pair_vectors[pair_key] = gate_vector
    return pair_vectors


def summarize_vectors(pair_vectors, common_keys):
    ts_values = []
    so_values = []
    t1c_values = []
    for pair_key in common_keys:
        _, _, type_a, type_b = pair_key
        gate_vector = pair_vectors[pair_key]
        ts_values.append(pair_target_sensitivity(gate_vector, type_a, type_b))
        so_values.append(pair_soft_overlap(gate_vector, type_a, type_b))
        t1c_values.append(pair_top1_collision(gate_vector, type_a, type_b))

    return {
        "TS": mean(ts_values),
        "SO": mean(so_values),
        "T1C": mean(t1c_values),
        "Pairs": len(ts_values),
    }


def fmt(value):
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    return "{:.4f}".format(value)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Model", "Run", "TS", "SO", "T1C", "Pairs"]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows):
    fields = ["Model", "TS", "SO", "T1C", "Pairs"]
    print("| " + " | ".join(fields) + " |")
    print("|" + "|".join(["---"] * len(fields)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[field]) for field in fields) + " |")


def main():
    parser = argparse.ArgumentParser(description="Export routing diagnostics from best_eval_dump.pkl.")
    parser.add_argument("--result-root", default="Result")
    parser.add_argument("--output", default="Result/routing_diagnostics.csv")
    parser.add_argument("--softmax-run", default="tgid_dueefin_softmax_t")
    parser.add_argument("--no-tc-run", default="tgid_dueefin_no_tc")
    parser.add_argument("--type-only-run", default="tgid_dueefin_type_only")
    parser.add_argument("--tgid-run", default="tgid_dueefin_sigmoid")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    runs = [
        ("TGID-Softmax", args.softmax_run),
        ("TGID w/o Ltc", args.no_tc_run),
        ("TGID type-only", args.type_only_run),
        ("TGID", args.tgid_run),
    ]

    rows = []
    pair_vectors_by_run = {}
    for model, run_name in runs:
        eval_dump = load_eval_dump(result_root, run_name)
        pair_vectors_by_run[run_name] = collect_pair_vectors(eval_dump) if eval_dump is not None else None

    available_key_sets = [
        set(pair_vectors.keys())
        for pair_vectors in pair_vectors_by_run.values()
        if pair_vectors is not None
    ]
    common_keys = set.intersection(*available_key_sets) if available_key_sets else set()

    for model, run_name in runs:
        pair_vectors = pair_vectors_by_run[run_name]
        summary = summarize_vectors(pair_vectors, common_keys) if pair_vectors is not None else None
        row = {"Model": model, "Run": run_name}
        if summary is None:
            row.update({"TS": "--", "SO": "--", "T1C": "--", "Pairs": "--"})
        else:
            row.update({key: fmt(value) for key, value in summary.items()})
        rows.append(row)

    write_csv(Path(args.output), rows)
    print_markdown(rows)
    print("\nCommon gold M2M target pairs: {}".format(len(common_keys)))
    print("\nWrote {}".format(args.output))


if __name__ == "__main__":
    main()
