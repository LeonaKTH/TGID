import argparse
import csv
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from procnet.conf.global_config_manager import GlobalConfigManager
from procnet.data_processor.DocEE_processor import DocEEProcessor


def sanitize_omp_num_threads():
    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_num_threads is None:
        return
    try:
        if int(omp_num_threads) > 0:
            return
    except ValueError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"


def event_type_set(event):
    return event.get("EventType")


def collect_argument_types(doc):
    arg_to_types = defaultdict(set)
    arg_to_roles = defaultdict(set)
    for event in doc.events:
        e_type = event_type_set(event)
        for role, value in event.items():
            if role == "EventType" or value is None:
                continue
            arg_to_types[value].add(e_type)
            arg_to_roles[value].add(role)
    return arg_to_types, arg_to_roles


def doc_has_cross_type_argument(doc):
    arg_to_types, _ = collect_argument_types(doc)
    return any(len(types) > 1 for types in arg_to_types.values())


def split_stats(name, docs):
    doc_num = len(docs)
    event_num = sum(len(doc.events) for doc in docs)
    multi_event_docs = sum(1 for doc in docs if len(doc.events) > 1)
    m2m_docs = 0
    total_args = 0
    cross_type_args = 0
    cross_type_role_pairs = Counter()
    cross_type_event_pairs = Counter()

    for doc in docs:
        arg_to_types, arg_to_roles = collect_argument_types(doc)
        has_m2m = False
        for _, types in arg_to_types.items():
            total_args += 1
            if len(types) > 1:
                has_m2m = True
                cross_type_args += 1
                cross_type_event_pairs.update([tuple(sorted(types))])
        for _, roles in arg_to_roles.items():
            if len(roles) > 1:
                cross_type_role_pairs.update([tuple(sorted(roles))])
        m2m_docs += int(has_m2m)

    return {
        "split": name,
        "doc_num": doc_num,
        "event_num": event_num,
        "multi_event_doc_num": multi_event_docs,
        "multi_event_doc_ratio": safe_div(multi_event_docs, doc_num),
        "m2m_doc_num": m2m_docs,
        "m2m_doc_ratio": safe_div(m2m_docs, doc_num),
        "argument_num": total_args,
        "cross_type_argument_num": cross_type_args,
        "cross_type_argument_ratio": safe_div(cross_type_args, total_args),
        "top_cross_type_event_pairs": counter_to_jsonable(cross_type_event_pairs, 10),
        "top_cross_role_pairs": counter_to_jsonable(cross_type_role_pairs, 10),
    }


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def counter_to_jsonable(counter, top_k):
    return [{"key": list(k), "count": v} for k, v in counter.most_common(top_k)]


def build_processor(data_name):
    return DocEEProcessor(data_name)


def run_data_stats(processor):
    return [
        split_stats("train", processor.train_docs),
        split_stats("dev", processor.dev_docs),
        split_stats("test", processor.test_docs),
    ]


def build_subset_doc_ids(processor, split):
    docs = getattr(processor, "{}_docs".format(split))
    m2m_ids = set()
    normal_ids = set()
    multi_event_ids = set()
    single_event_ids = set()
    for doc in docs:
        if doc_has_cross_type_argument(doc):
            m2m_ids.add(doc.doc_id)
        else:
            normal_ids.add(doc.doc_id)
        if len(doc.events) > 1:
            multi_event_ids.add(doc.doc_id)
        else:
            single_event_ids.add(doc.doc_id)
    return {
        "m2m": m2m_ids,
        "non_m2m": normal_ids,
        "multi_event": multi_event_ids,
        "single_event": single_event_ids,
    }


def subset_f1_from_raw(raw_result_path, subset_ids, preparer):
    import torch
    from procnet.metric.DocEE_metric import DocEEMetric

    raw_results = torch.load(raw_result_path, map_location="cpu")
    metric = DocEEMetric(preparer=preparer)
    output = {}
    for name, doc_ids in subset_ids.items():
        subset = [res for res in raw_results if res["doc_id"] in doc_ids]
        if not subset:
            output[name] = {"doc_num": 0, "event": {}}
            continue
        _, score = metric.the_score_fn(subset)
        output[name] = {
            "doc_num": len(subset),
            "event": score.get("event", {}),
        }
    return output


class ProxyCapture:
    def __init__(self):
        self.features = None
        self.logits = None
        self.handles = []

    def attach(self, model):
        def input_hook(module, module_input, module_output):
            self.features = module_input[0].detach().cpu()
            self.logits = module_output.detach().cpu()

        self.handles.append(model.proxy_slot_event_type_linear.register_forward_hook(input_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def load_model_for_analysis(config, preparer, checkpoint):
    import torch
    from procnet.model.DocEE_proxy_node_model import DocEEProxyNodeModel
    from run import resolve_checkpoint_path

    checkpoint_path = resolve_checkpoint_path(checkpoint)
    model = torch.load(checkpoint_path, map_location=config.device)
    if not isinstance(model, DocEEProxyNodeModel):
        logging.warning("Loaded checkpoint type is %s; continuing if it exposes EPAL proxy modules.", type(model))
    model.to(config.device)
    model.device = config.device
    if hasattr(model, "pos"):
        model.pos = model.pos.to(config.device)
    model.eval()
    return model


def extract_proxy_representations(model, loader, m2m_doc_ids, true_bio=False):
    import torch
    from tqdm import tqdm

    capture = ProxyCapture()
    capture.attach(model)
    rows = []
    feature_rows = []
    with torch.no_grad():
        for example in tqdm(loader, desc="Extracting proxy representations"):
            capture.features = None
            capture.logits = None
            bios_ids = [x.to(model.device) for x in example.BIO_ids] if true_bio else None
            _, result = model(example=example, bios_ids=bios_ids, use_mix_bio=False)
            if capture.features is None or capture.logits is None:
                continue
            if len(example.events_label) == 0:
                continue

            logits = capture.logits.numpy()
            features = capture.features.numpy()
            doc_m2m = example.doc_id in m2m_doc_ids
            used_proxy = set()
            for event_index, event_label in enumerate(example.events_label):
                label = int(event_label["EventType"])
                order = np.argsort(logits[:, label])[::-1]
                chosen = int(order[0])
                for idx in order:
                    if int(idx) not in used_proxy:
                        chosen = int(idx)
                        break
                used_proxy.add(chosen)
                feature_rows.append(features[chosen])
                rows.append(
                    {
                        "doc_id": example.doc_id,
                        "event_index": event_index,
                        "event_type_id": label,
                        "proxy_index": chosen,
                        "proxy_score": float(logits[chosen, label]),
                        "doc_is_m2m": int(doc_m2m),
                        "loss": float(result.get("loss", 0.0)),
                    }
                )
    capture.close()
    if feature_rows:
        feature_matrix = np.stack(feature_rows)
    else:
        feature_matrix = np.zeros((0, 0), dtype=np.float32)
    return rows, feature_matrix


def cosine_summary(features, labels):
    if len(features) < 2:
        return {}
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    z = features / norm
    sim = z @ z.T
    same = []
    diff = []
    for i in range(len(labels)):
        for j in range(i):
            if labels[i] == labels[j]:
                same.append(sim[i, j])
            else:
                diff.append(sim[i, j])
    return {
        "same_type_cosine_mean": float(np.mean(same)) if same else 0.0,
        "same_type_pair_num": len(same),
        "diff_type_cosine_mean": float(np.mean(diff)) if diff else 0.0,
        "diff_type_pair_num": len(diff),
        "cosine_margin_same_minus_diff": float(np.mean(same) - np.mean(diff)) if same and diff else 0.0,
    }


def linear_probe_summary(features, labels):
    if len(features) < 4 or len(set(labels)) < 2:
        return {"available": False, "reason": "not enough labeled representations"}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except Exception as exc:
        return {"available": False, "reason": "scikit-learn is unavailable: {}".format(exc)}

    min_class_count = min(Counter(labels).values())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        return {"available": False, "reason": "each event type needs at least two samples"}

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13)
    scores = cross_val_score(clf, features, labels, cv=cv, scoring="f1_macro")
    return {
        "available": True,
        "cv_f1_macro_mean": float(np.mean(scores)),
        "cv_f1_macro_std": float(np.std(scores)),
        "cv_splits": n_splits,
        "sample_num": int(len(labels)),
        "class_num": int(len(set(labels))),
    }


def maybe_tsne(features, rows, output_dir):
    if len(features) < 3:
        return None
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return None
    perplexity = min(30, max(2, len(features) // 3))
    if perplexity >= len(features):
        perplexity = len(features) - 1
    points = TSNE(n_components=2, random_state=13, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(features)
    tsne_path = output_dir / "proxy_tsne.csv"
    with open(tsne_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["x", "y"] + list(rows[0].keys()))
        writer.writeheader()
        for point, row in zip(points, rows):
            out = {"x": float(point[0]), "y": float(point[1])}
            out.update(row)
            writer.writerow(out)
    return str(tsne_path)


def save_rows(rows, features, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "proxy_representations.csv"
    if rows:
        with open(rows_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    np.save(output_dir / "proxy_features.npy", features)
    return str(rows_path), str(output_dir / "proxy_features.npy")


def build_preparer_and_loader(config, processor, split):
    from procnet.data_preparer.DocEE_preparer import DocEEPreparer

    preparer = DocEEPreparer(config=config, processor=processor)
    train_loader, dev_loader, test_loader = preparer.get_loader_for_flattened_fragment_before_event()
    loaders = {"train": train_loader, "dev": dev_loader, "test": test_loader}
    return preparer, loaders[split]


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 0 for EPAL: M2M statistics, subset F1, and representation separability.")
    parser.add_argument("--data-name", default="ChFinAnn", help="Dataset name under Data/.")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--raw-result", default=None, help="Path to raw.pt from EPAL evaluation for subset F1 analysis.")
    parser.add_argument("--checkpoint", default=None, help="EPAL checkpoint for representation separability analysis.")
    parser.add_argument("--true-bio", action="store_true", help="Use gold BIO spans when extracting representations.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda.")
    parser.add_argument("--output-dir", default="Result/experiment0_epal")
    return parser.parse_args()


def main():
    sanitize_omp_num_threads()
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO, datefmt="%I:%M:%S")
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = GlobalConfigManager.current_path / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = build_processor(args.data_name)
    subset_ids = build_subset_doc_ids(processor, args.split)
    summary = {
        "data_name": args.data_name,
        "split": args.split,
        "data_stats": run_data_stats(processor),
    }

    preparer = None
    loader = None
    if args.raw_result is not None or args.checkpoint is not None:
        import torch
        from run import get_config

        config = get_config()
        config.data_name = args.data_name
        if args.device is not None:
            config.device = torch.device(args.device)
        preparer, loader = build_preparer_and_loader(config, processor, args.split)

    if args.raw_result is not None:
        summary["subset_f1"] = subset_f1_from_raw(args.raw_result, subset_ids, preparer)

    if args.checkpoint is not None:
        model = load_model_for_analysis(config, preparer, args.checkpoint)
        rows, features = extract_proxy_representations(model, loader, subset_ids["m2m"], true_bio=args.true_bio)
        labels = [row["event_type_id"] for row in rows]
        rows_path, features_path = save_rows(rows, features, output_dir)
        summary["representation"] = {
            "row_path": rows_path,
            "feature_path": features_path,
            "sample_num": len(rows),
            "event_type_counts": dict(Counter(labels)),
            "cosine": cosine_summary(features, labels),
            "linear_probe": linear_probe_summary(features, labels),
            "tsne_path": maybe_tsne(features, rows, output_dir) if rows else None,
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logging.info("Experiment 0 summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
