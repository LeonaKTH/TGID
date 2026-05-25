import argparse
from pathlib import Path

import torch

TOKENIZER = None

ALIAS_TOKENS = [
    "[A]", "[B]", "[C]", "[D]", "[E]", "[F]", "[G]", "[H]", "[I]", "[J]",
    "[K]", "[L]", "[M]", "[N]", "[O]", "[P]", "[Q]", "[R]", "[S]", "[T]",
]


def load_tokenizer(name):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:
        print("Warning: failed to load tokenizer '{}': {}".format(name, exc))
        return None


def normalize_mapping(mapping):
    if isinstance(mapping, dict):
        output = {}
        for key, value in mapping.items():
            try:
                output[int(key)] = value
            except (TypeError, ValueError):
                output[key] = value
        return output
    if isinstance(mapping, list):
        return {idx: value for idx, value in enumerate(mapping)}
    return {}


def arg_text(value):
    if value is None:
        return "-"
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        if TOKENIZER is not None and all(isinstance(item, int) for item in value):
            tokens = TOKENIZER.convert_ids_to_tokens(list(value))
            text = TOKENIZER.convert_tokens_to_string(tokens)
            return text.replace(" ", "")
        return str(tuple(value))
    if isinstance(value, int) and TOKENIZER is not None:
        return TOKENIZER.convert_tokens_to_string(TOKENIZER.convert_ids_to_tokens([value])).replace(" ", "")
    return str(value)


def best_index(values):
    best_value = None
    best_idx = 0
    for idx, value in enumerate(values):
        if best_value is None or value > best_value:
            best_value = value
            best_idx = idx
    return best_idx


def decode_gold_events(event_ans, schema, type_index_to_name, role_name_to_index):
    decoded = []
    for instance_idx, event in enumerate(event_ans):
        event_type = type_index_to_name.get(int(event["EventType"]), str(event["EventType"]))
        role_values = {}
        for role_name in schema.get(event_type, []):
            role_index = role_name_to_index.get(role_name)
            chosen = None
            for key, role_ids in event.items():
                if key == "EventType":
                    continue
                values = role_ids if isinstance(role_ids, list) else [role_ids]
                if role_index in values:
                    chosen = key
                    break
            role_values[role_name] = chosen
        decoded.append({"event_type": event_type, "instance": instance_idx + 1, "roles": role_values})
    return decoded


def decode_pred_events(event_pred, schema, type_index_to_name):
    if not event_pred:
        return []
    decoded = []
    none_index = len(event_pred.get("index", []))
    for instance_idx, type_scores in enumerate(event_pred.get("type", [])):
        event_type_id = best_index(type_scores)
        event_type = type_index_to_name.get(int(event_type_id), str(event_type_id))
        if event_type == "Null":
            continue
        role_values = {}
        rel_scores = event_pred.get("rel", [])[instance_idx]
        for role_idx, role_name in enumerate(schema.get(event_type, [])):
            if role_idx >= len(rel_scores):
                role_values[role_name] = None
                continue
            arg_idx = best_index(rel_scores[role_idx])
            if arg_idx == none_index or arg_idx >= len(event_pred.get("index", [])):
                role_values[role_name] = None
            else:
                role_values[role_name] = event_pred["index"][arg_idx]
        decoded.append({"event_type": event_type, "instance": instance_idx + 1, "roles": role_values})
    return decoded


def match_pred(gold_event, pred_events):
    same_type = [event for event in pred_events if event["event_type"] == gold_event["event_type"]]
    if not same_type:
        return None
    return max(
        same_type,
        key=lambda pred: sum(
            1
            for role, gold_value in gold_event["roles"].items()
            if gold_value is not None and pred["roles"].get(role) == gold_value
        ),
    )


def role_correct_count(gold_events, pred_events):
    total = 0
    correct = 0
    event_correct = []
    for gold_event in gold_events:
        pred = match_pred(gold_event, pred_events)
        event_total = 0
        event_ok = 0
        for _, gold_value in gold_event["roles"].items():
            if gold_value is None:
                continue
            total += 1
            event_total += 1
            if pred is not None and gold_value in pred["roles"].values():
                correct += 1
                event_ok += 1
        event_correct.append((event_ok, event_total))
    return correct, total, event_correct


def has_later_event_gap(event_correct):
    if len(event_correct) < 2:
        return False
    first_ok, _ = event_correct[0]
    return first_ok > 0 and any(ok == 0 and total > 0 for ok, total in event_correct[1:])


def distinct_filled_roles(gold_events):
    roles = set()
    for event in gold_events:
        for role, value in event["roles"].items():
            if value is not None:
                roles.add(role)
    return roles


def score_doc(gold_events, epal_events, softmax_events, tgid_events):
    epal_correct, total, epal_event_correct = role_correct_count(gold_events, epal_events)
    softmax_correct, _, _ = role_correct_count(gold_events, softmax_events)
    tgid_correct, _, _ = role_correct_count(gold_events, tgid_events)
    if total == 0:
        return 0
    score = (tgid_correct - epal_correct) + (tgid_correct - softmax_correct)
    if has_later_event_gap(epal_event_correct):
        score += 2
    if tgid_correct == total:
        score += 2
    return score


def shared_cross_type_arguments(gold_events):
    arg_to_event_types = {}
    for event in gold_events:
        for value in event["roles"].values():
            if value is None:
                continue
            key = arg_text(value)
            arg_to_event_types.setdefault(key, set()).add(event["event_type"])
    return {key for key, event_types in arg_to_event_types.items() if len(event_types) >= 2}


def cross_type_score_doc(gold_events, epal_events, softmax_events, tgid_events):
    shared_args = shared_cross_type_arguments(gold_events)
    if not shared_args:
        return 0
    return score_doc(gold_events, epal_events, softmax_events, tgid_events) + len(shared_args)


def load_dump(path):
    return torch.load(path, map_location="cpu")


def index_raw_results(eval_dump):
    return {str(row["doc_id"]): row for row in eval_dump.get("raw_results", [])}


def latex_escape(text):
    text = str(text)
    for src, dst in [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
    ]:
        text = text.replace(src, dst)
    return text


def value_with_mark(value, ok):
    if value is None:
        return "-- $\\times$"
    mark = "$\\checkmark$" if ok else "$\\times$"
    return "{} {}".format(latex_escape(arg_text(value)), mark)


def anonymizer(enabled):
    mapping = {}

    def convert(value):
        if value is None:
            return None
        key = arg_text(value)
        if not enabled:
            return latex_escape(key)
        if key not in mapping:
            mapping[key] = ALIAS_TOKENS[len(mapping)] if len(mapping) < len(ALIAS_TOKENS) else "[X{}]".format(len(mapping) + 1)
        return mapping[key]

    return convert, mapping


def value_with_mark_alias(value, ok, alias_fn):
    alias = alias_fn(value)
    if alias is None:
        return "-- $\\times$"
    mark = "$\\checkmark$" if ok else "$\\times$"
    return "{} {}".format(alias, mark)


def build_rows(gold_events, epal_events, softmax_events, tgid_events, max_rows, require_cross_type=False, anonymize=False):
    rows = []
    shared_args = shared_cross_type_arguments(gold_events) if require_cross_type else None
    alias_fn, alias_mapping = anonymizer(anonymize)
    for gold_event in gold_events:
        epal = match_pred(gold_event, epal_events)
        softmax = match_pred(gold_event, softmax_events)
        tgid = match_pred(gold_event, tgid_events)
        for role, gold_value in gold_event["roles"].items():
            if gold_value is None:
                continue
            epal_value = epal["roles"].get(role) if epal is not None else None
            softmax_value = softmax["roles"].get(role) if softmax is not None else None
            tgid_value = tgid["roles"].get(role) if tgid is not None else None
            epal_ok = epal_value == gold_value
            softmax_ok = softmax_value == gold_value
            tgid_ok = tgid_value == gold_value
            shared_cross_type = shared_args is not None and arg_text(gold_value) in shared_args
            informative = tgid_ok and (not epal_ok or not softmax_ok)
            if shared_args is not None and not (shared_cross_type or informative):
                continue
            rows.append(
                [
                    "{} {} / {}".format(gold_event["event_type"], gold_event["instance"], role),
                    alias_fn(gold_value),
                    value_with_mark_alias(epal_value, epal_ok, alias_fn),
                    value_with_mark_alias(softmax_value, softmax_ok, alias_fn),
                    value_with_mark_alias(tgid_value, tgid_ok, alias_fn),
                ]
            )
            if len(rows) >= max_rows:
                return rows, alias_mapping
    return rows, alias_mapping


def event_distribution(gold_events):
    dist = {}
    for event in gold_events:
        dist[event["event_type"]] = dist.get(event["event_type"], 0) + 1
    return dist


def write_latex(path, cases, caption):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for case_idx, case in enumerate(cases, start=1):
            file.write("% Case {}: doc_id={}, event_distribution={}\n".format(
                case_idx, case["doc_id"], case["event_distribution"]
            ))
            file.write("\\begin{table}[t]\n")
            file.write("\\centering\n")
            file.write("\\small\n")
            file.write("\\caption{" + caption + " Candidate " + str(case_idx) + ".}\n")
            file.write("\\begin{tabular}{lllll}\n")
            file.write("\\toprule\n")
            file.write("Event / Role & Gold & EPAL & TGID-Softmax & TGID \\\\\n")
            file.write("\\midrule\n")
            for row in case["rows"]:
                file.write(" & ".join(row) + " \\\\\n")
            file.write("\\bottomrule\n")
            file.write("\\end{tabular}\n")
            file.write("\\end{table}\n\n")


def write_markdown(path, cases):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for case_idx, case in enumerate(cases, start=1):
            file.write("### Candidate {}: {}\n\n".format(case_idx, case["doc_id"]))
            file.write("Score: {}\n\n".format(case["score"]))
            file.write("Event distribution: `{}`\n\n".format(case["event_distribution"]))
            file.write("| Event / Role | Gold | EPAL | TGID-Softmax | TGID |\n")
            file.write("|---|---|---|---|---|\n")
            for row in case["rows"]:
                file.write("| {} |\n".format(" | ".join(row)))
            file.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Export EPAL-vs-TGID qualitative case table.")
    parser.add_argument("--epal-dump", default="Result/epal_dueefin_eval/best_eval_dump.pkl")
    parser.add_argument("--softmax-dump", default="Result/tgid_dueefin_softmax_t/best_eval_dump.pkl")
    parser.add_argument("--tgid-dump", default="Result/tgid_dueefin_sigmoid/best_eval_dump.pkl")
    parser.add_argument("--output", default="figures/case_study_table.tex")
    parser.add_argument("--output-md", default="figures/case_study_candidates.md")
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--event-type-filter", default=None)
    parser.add_argument("--min-event-types", type=int, default=2)
    parser.add_argument("--min-distinct-roles", type=int, default=2)
    parser.add_argument("--require-epal-later-gap", action="store_true")
    parser.add_argument("--require-softmax-error", action="store_true")
    parser.add_argument("--require-tgid-perfect", action="store_true")
    parser.add_argument("--anonymize", action="store_true")
    parser.add_argument("--allow-non-cross-type", action="store_true")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--doc-id", default=None)
    args = parser.parse_args()

    epal_dump = load_dump(Path(args.epal_dump))
    softmax_dump = load_dump(Path(args.softmax_dump))
    tgid_dump = load_dump(Path(args.tgid_dump))
    global TOKENIZER
    tokenizer_name = args.tokenizer_name or tgid_dump.get("config", {}).get("model_name", "hfl/chinese-roberta-wwm-ext")
    TOKENIZER = load_tokenizer(tokenizer_name)
    epal_raw = index_raw_results(epal_dump)
    softmax_raw = index_raw_results(softmax_dump)
    tgid_raw = index_raw_results(tgid_dump)
    common_doc_ids = sorted(set(epal_raw) & set(softmax_raw) & set(tgid_raw))
    schema = tgid_dump.get("schema", {})
    type_index_to_name = normalize_mapping(tgid_dump.get("event_type_index_to_type", {}))
    role_name_to_index = tgid_dump.get("event_role_relation_to_index", {})

    candidates = []
    for doc_id in common_doc_ids:
        if args.doc_id is not None and str(doc_id) != str(args.doc_id):
            continue
        gold_events = decode_gold_events(tgid_raw[doc_id]["event_ans"], schema, type_index_to_name, role_name_to_index)
        if len(gold_events) < 2:
            continue
        dist = event_distribution(gold_events)
        if len(dist) < args.min_event_types:
            continue
        if len(distinct_filled_roles(gold_events)) < args.min_distinct_roles:
            continue
        epal_events = decode_pred_events(epal_raw[doc_id]["event_pred"], schema, type_index_to_name)
        softmax_events = decode_pred_events(softmax_raw[doc_id]["event_pred"], schema, type_index_to_name)
        tgid_events = decode_pred_events(tgid_raw[doc_id]["event_pred"], schema, type_index_to_name)
        epal_correct, total_roles, epal_event_correct = role_correct_count(gold_events, epal_events)
        softmax_correct, _, _ = role_correct_count(gold_events, softmax_events)
        tgid_correct, _, _ = role_correct_count(gold_events, tgid_events)
        if total_roles == 0:
            continue
        if args.require_epal_later_gap and not has_later_event_gap(epal_event_correct):
            continue
        if args.require_softmax_error and softmax_correct == total_roles:
            continue
        if args.require_tgid_perfect and tgid_correct != total_roles:
            continue
        score = cross_type_score_doc(gold_events, epal_events, softmax_events, tgid_events)
        if score <= 0 and args.allow_non_cross_type:
            score = score_doc(gold_events, epal_events, softmax_events, tgid_events)
        if score <= 0:
            continue
        rows, alias_mapping = build_rows(
            gold_events,
            epal_events,
            softmax_events,
            tgid_events,
            args.max_rows,
            require_cross_type=not args.allow_non_cross_type,
            anonymize=args.anonymize,
        )
        if not rows:
            continue
        if args.event_type_filter is not None and args.event_type_filter not in dist:
            continue
        if len(rows) < args.min_rows:
            continue
        candidates.append(
            {
                "doc_id": doc_id,
                "score": score,
                "epal_correct": epal_correct,
                "softmax_correct": softmax_correct,
                "tgid_correct": tgid_correct,
                "total_roles": total_roles,
                "gold_events": gold_events,
                "epal_events": epal_events,
                "softmax_events": softmax_events,
                "tgid_events": tgid_events,
                "rows": rows,
                "alias_mapping": alias_mapping,
                "event_distribution": dist,
            }
        )

    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[: args.num_cases]
    if not candidates:
        raise RuntimeError("No case found where TGID fixes EPAL errors.")
    caption = (
        "Qualitative example from a Cross-Type M2M document. Only representative roles are shown. "
        "A check mark indicates an exactly matched argument, and a cross indicates an incorrect or missing argument."
    )
    write_latex(Path(args.output), candidates, caption)
    write_markdown(Path(args.output_md) if args.output_md else None, candidates)
    for index, case in enumerate(candidates, start=1):
        print("Candidate {}: doc_id={}, score={}, correct EPAL/Softmax/TGID={}/{}/{}, total={}, event_distribution={}".format(
            index,
            case["doc_id"],
            case["score"],
            case["epal_correct"],
            case["softmax_correct"],
            case["tgid_correct"],
            case["total_roles"],
            case["event_distribution"],
        ))
        if args.anonymize:
            print("  Argument aliases:")
            for raw, alias in sorted(case["alias_mapping"].items(), key=lambda item: item[1]):
                print("    {} = {}".format(alias, raw))
    print("Wrote {}".format(args.output))
    if args.output_md:
        print("Wrote {}".format(args.output_md))


if __name__ == "__main__":
    main()
