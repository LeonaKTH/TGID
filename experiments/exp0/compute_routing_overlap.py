import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from utils import classify_doc_arguments, normalize_docs, read_json, safe_div, write_json


def tensorize_route_prob(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float)


def topk_set(prob, top_k):
    k = min(top_k, prob.shape[-1])
    return set(torch.topk(prob, k=k, dim=-1).indices.cpu().numpy().tolist())


def jaccard(a, b):
    union = a | b
    return safe_div(len(a & b), len(union))


def pair_overlap(route_prob, type_a, type_b, top_k):
    return jaccard(topk_set(route_prob[type_a], top_k), topk_set(route_prob[type_b], top_k))


def pair_target_sensitivity(route_prob, type_a, type_b):
    return float(1.0 - torch.cosine_similarity(route_prob[type_a], route_prob[type_b], dim=0).item())


def pair_soft_overlap(route_prob, type_a, type_b):
    return float(torch.minimum(route_prob[type_a], route_prob[type_b]).sum().item())


def pair_top1_collision(route_prob, type_a, type_b):
    return float(torch.argmax(route_prob[type_a]).item() == torch.argmax(route_prob[type_b]).item())


def summarize(values, key="mean_overlap"):
    return {key: float(sum(values) / len(values)) if values else 0.0, "num_type_pairs": len(values)}


def bucket_ratio(value):
    if value == 0:
        return "0"
    if value <= 0.1:
        return "(0,0.1]"
    if value <= 0.3:
        return "(0.1,0.3]"
    if value <= 0.5:
        return "(0.3,0.5]"
    return "(0.5,1]"


def load_doc_ratios(gold_data):
    if not gold_data:
        return {}
    docs = normalize_docs(read_json(gold_data))
    ratios = {}
    for doc in docs:
        arg_info = classify_doc_arguments(doc)
        cross = sum(1 for x in arg_info.values() if x["category"] == "CROSS_TYPE_M2M")
        ratios[doc["doc_id"]] = safe_div(cross, len(arg_info))
    return ratios


def main():
    parser = argparse.ArgumentParser(description="Experiment 0 Routing-Overlap and Route Entropy.")
    parser.add_argument("--route_dump", required=True)
    parser.add_argument("--gold_data", default=None)
    parser.add_argument("--output", default="outputs/exp0/routing_overlap.json")
    parser.add_argument("--entropy_output", default=None)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=13)
    args = parser.parse_args()

    records = torch.load(args.route_dump, map_location="cpu")
    doc_ratios = load_doc_ratios(args.gold_data)
    random.seed(args.random_seed)

    by_category = defaultdict(list)
    sensitivity_by_category = defaultdict(list)
    soft_overlap_by_category = defaultdict(list)
    top1_collision_by_category = defaultdict(list)
    by_type_pair = defaultdict(list)
    by_bucket = defaultdict(list)
    random_overlaps = []
    entropy_by_category = defaultdict(list)

    for record in records:
        route_prob = tensorize_route_prob(record["route_prob"])
        category = record.get("category", "SINGLE")
        gold_type_ids = [int(x) for x in record.get("gold_event_type_ids", [])]

        for type_id in gold_type_ids:
            if 0 <= type_id < route_prob.shape[0]:
                p = route_prob[type_id].clamp_min(1e-12)
                entropy_by_category[category].append(float(-(p * torch.log(p)).sum().item()))

        if len(gold_type_ids) >= 2:
            for type_a, type_b in itertools.combinations(sorted(gold_type_ids), 2):
                if type_a >= route_prob.shape[0] or type_b >= route_prob.shape[0]:
                    continue
                overlap = pair_overlap(route_prob, type_a, type_b, args.top_k)
                by_category[category].append(overlap)
                by_type_pair["{}|{}".format(type_a, type_b)].append(overlap)
                doc_id = str(record.get("doc_id", ""))
                by_bucket[bucket_ratio(doc_ratios.get(doc_id, 0.0))].append(overlap)
                sensitivity_by_category[category].append(pair_target_sensitivity(route_prob, type_a, type_b))
                soft_overlap_by_category[category].append(pair_soft_overlap(route_prob, type_a, type_b))
                top1_collision_by_category[category].append(pair_top1_collision(route_prob, type_a, type_b))

        valid_types = list(range(1, route_prob.shape[0])) or list(range(route_prob.shape[0]))
        if len(valid_types) >= 2:
            type_a, type_b = random.sample(valid_types, 2)
            random_overlaps.append(pair_overlap(route_prob, type_a, type_b, args.top_k))

    overlap_output = {
        "top_k": args.top_k,
        "cross_type_m2m": {
            **summarize(by_category.get("CROSS_TYPE_M2M", [])),
            "num_arguments": sum(1 for r in records if r.get("category") == "CROSS_TYPE_M2M"),
        },
        "same_type_multi": {
            **summarize(by_category.get("SAME_TYPE_MULTI", [])),
            "num_arguments": sum(1 for r in records if r.get("category") == "SAME_TYPE_MULTI"),
        },
        "target_sensitivity": {
            "cross_type_m2m": summarize(
                sensitivity_by_category.get("CROSS_TYPE_M2M", []),
                key="mean_target_sensitivity",
            ),
            "same_type_multi": summarize(
                sensitivity_by_category.get("SAME_TYPE_MULTI", []),
                key="mean_target_sensitivity",
            ),
        },
        "soft_overlap": {
            "cross_type_m2m": summarize(
                soft_overlap_by_category.get("CROSS_TYPE_M2M", []),
                key="mean_soft_overlap",
            ),
            "same_type_multi": summarize(
                soft_overlap_by_category.get("SAME_TYPE_MULTI", []),
                key="mean_soft_overlap",
            ),
        },
        "top1_collision": {
            "cross_type_m2m": summarize(
                top1_collision_by_category.get("CROSS_TYPE_M2M", []),
                key="mean_top1_collision",
            ),
            "same_type_multi": summarize(
                top1_collision_by_category.get("SAME_TYPE_MULTI", []),
                key="mean_top1_collision",
            ),
        },
        "random_type_pairs": {
            "mean_overlap": float(sum(random_overlaps) / len(random_overlaps)) if random_overlaps else 0.0,
            "num_pairs": len(random_overlaps),
        },
        "by_type_pair": {key: summarize(values) for key, values in sorted(by_type_pair.items())},
        "by_doc_m2m_ratio_bucket": {key: summarize(values) for key, values in sorted(by_bucket.items())},
    }
    write_json(args.output, overlap_output)

    entropy_output = {}
    for name in ["SINGLE", "SAME_TYPE_MULTI", "CROSS_TYPE_M2M"]:
        values = entropy_by_category.get(name, [])
        entropy_output[name.lower()] = {
            "mean_entropy": float(sum(values) / len(values)) if values else 0.0,
            "num_items": len(values),
        }
    entropy_path = args.entropy_output
    if entropy_path is None:
        entropy_path = str(Path(args.output).with_name("route_entropy.json"))
    write_json(entropy_path, entropy_output)


if __name__ == "__main__":
    main()
