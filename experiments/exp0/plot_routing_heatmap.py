import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
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


def load_dump(result_root, run_name):
    return torch.load(result_root / run_name / "best_eval_dump.pkl", map_location="cpu")


def gate_name(eval_dump):
    return str(eval_dump.get("config", {}).get("tgid_gate", "sigmoid"))


def continuous_gate(record, gate):
    if record.get("route_logits") is not None:
        logits = tensorize(record["route_logits"])
        if gate == "softmax":
            return F.softmax(logits, dim=-1)
        score = torch.sigmoid(logits)
        return score / score.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    prob = tensorize(record["route_prob"])
    return prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def route_index(eval_dump):
    gate = gate_name(eval_dump)
    output = {}
    for record in eval_dump.get("route_records", []):
        if not record.get("is_cross_type", False):
            continue
        gold_type_ids = sorted({int(x) for x in record.get("gold_event_type_ids", [])})
        if len(gold_type_ids) < 2:
            continue
        doc_id = str(record.get("doc_id", ""))
        arg_key = stringify_key(record.get("arg_key", ""))
        output[(doc_id, arg_key)] = {
            "record": record,
            "gold_type_ids": gold_type_ids,
            "gate": continuous_gate(record, gate),
        }
    return output


def cosine_distance(gate, type_a, type_b):
    return float(1.0 - F.cosine_similarity(gate[type_a], gate[type_b], dim=0).item())


def choose_case(softmax_routes, tgid_routes):
    common = sorted(set(softmax_routes) & set(tgid_routes))
    best = None
    for key in common:
        soft = softmax_routes[key]
        tgid = tgid_routes[key]
        common_types = sorted(set(soft["gold_type_ids"]) & set(tgid["gold_type_ids"]))
        if len(common_types) < 2:
            continue
        type_a, type_b = common_types[0], common_types[1]
        soft_ts = cosine_distance(soft["gate"], type_a, type_b)
        tgid_ts = cosine_distance(tgid["gate"], type_a, type_b)
        score = tgid_ts - soft_ts
        if best is None or score > best["score"]:
            best = {
                "key": key,
                "type_a": type_a,
                "type_b": type_b,
                "soft_ts": soft_ts,
                "tgid_ts": tgid_ts,
                "score": score,
            }
    return best


def target_label(type_id, event_type_index_to_type):
    if isinstance(event_type_index_to_type, dict):
        return str(event_type_index_to_type.get(str(type_id), event_type_index_to_type.get(type_id, type_id)))
    if isinstance(event_type_index_to_type, list) and type_id < len(event_type_index_to_type):
        return str(event_type_index_to_type[type_id])
    return "Type {}".format(type_id)


def write_csv(path, rows, num_experts):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "argument", "target"] + ["E{}".format(i + 1) for i in range(num_experts)]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(path, soft_rows, tgid_rows, expert_labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.6), constrained_layout=True)
    for ax, title, rows in [
        (axes[0], "(a) TGID-Softmax", soft_rows),
        (axes[1], "(b) TGID", tgid_rows),
    ]:
        matrix = torch.stack([row["values"] for row in rows]).numpy()
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(0.35, float(matrix.max())))
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(expert_labels)))
        ax.set_xticklabels(expert_labels, fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row["target"] for row in rows], fontsize=8)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                ax.text(x, y, "{:.2f}".format(matrix[y, x]), ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Gate activation")
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a routing heatmap case study.")
    parser.add_argument("--result-root", default="Result")
    parser.add_argument("--softmax-run", default="tgid_dueefin_softmax_t")
    parser.add_argument("--tgid-run", default="tgid_dueefin_sigmoid")
    parser.add_argument("--output-pdf", default="figures/routing_heatmap.pdf")
    parser.add_argument("--output-png", default="figures/routing_heatmap.png")
    parser.add_argument("--output-csv", default="figures/routing_heatmap.csv")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    softmax_dump = load_dump(result_root, args.softmax_run)
    tgid_dump = load_dump(result_root, args.tgid_run)
    softmax_routes = route_index(softmax_dump)
    tgid_routes = route_index(tgid_dump)
    case = choose_case(softmax_routes, tgid_routes)
    if case is None:
        raise RuntimeError("No shared Cross-Type M2M case found.")

    doc_id, arg_key = case["key"]
    type_a, type_b = case["type_a"], case["type_b"]
    event_types = tgid_dump.get("event_type_index_to_type", [])
    targets = [target_label(type_a, event_types), target_label(type_b, event_types)]
    soft_gate = softmax_routes[case["key"]]["gate"]
    tgid_gate = tgid_routes[case["key"]]["gate"]
    num_experts = soft_gate.shape[-1]
    expert_labels = ["E{}".format(i + 1) for i in range(num_experts)]

    soft_plot_rows = [
        {"target": targets[0], "values": soft_gate[type_a]},
        {"target": targets[1], "values": soft_gate[type_b]},
    ]
    tgid_plot_rows = [
        {"target": targets[0], "values": tgid_gate[type_a]},
        {"target": targets[1], "values": tgid_gate[type_b]},
    ]
    csv_rows = []
    for model, rows in [("TGID-Softmax", soft_plot_rows), ("TGID", tgid_plot_rows)]:
        for row in rows:
            csv_row = {"model": model, "argument": arg_key, "target": row["target"]}
            csv_row.update({"E{}".format(i + 1): "{:.6f}".format(float(v)) for i, v in enumerate(row["values"])})
            csv_rows.append(csv_row)

    write_csv(Path(args.output_csv), csv_rows, num_experts)
    plot_heatmap(Path(args.output_pdf), soft_plot_rows, tgid_plot_rows, expert_labels)
    plot_heatmap(Path(args.output_png), soft_plot_rows, tgid_plot_rows, expert_labels)
    print("Selected doc_id={}, argument={}, targets={} / {}".format(doc_id, arg_key, targets[0], targets[1]))
    print("TS softmax={:.4f}, TGID={:.4f}, margin={:.4f}".format(case["soft_ts"], case["tgid_ts"], case["score"]))
    print("Wrote {}, {}, {}".format(args.output_pdf, args.output_png, args.output_csv))


if __name__ == "__main__":
    main()
