import json
from collections import defaultdict
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def normalize_docs(raw):
    if isinstance(raw, dict) and "documents" in raw:
        raw = raw["documents"]
    docs = []
    for idx, item in enumerate(raw):
        if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], dict):
            doc_id = str(item[0])
            data = item[1]
            events = []
            for event_item in data.get("recguid_eventname_eventdict_list", []):
                event_type = event_item[1]
                event_dict = event_item[2]
                event = {"EventType": event_type}
                event.update(event_dict)
                events.append(event)
            arg_positions = defaultdict(list)
            for text, positions in data.get("ann_mspan2dranges", {}).items():
                for pos in positions:
                    if len(pos) >= 3:
                        arg_positions[text].append((int(pos[1]), int(pos[2]), text))
            docs.append({"doc_id": doc_id, "events": events, "arg_positions": dict(arg_positions)})
        elif isinstance(item, dict):
            doc_id = str(item.get("doc_id", item.get("id", idx)))
            events = item.get("events", item.get("event_list", []))
            normalized_events = []
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_type = event.get("EventType", event.get("event_type", event.get("type")))
                normalized = {"EventType": event_type}
                for key, value in event.items():
                    if key in {"EventType", "event_type", "type", "arguments", "argument_list"}:
                        continue
                    normalized[key] = value
                for arg in event.get("arguments", event.get("argument_list", [])):
                    role = arg.get("role")
                    if role:
                        normalized[role] = arg.get("text", arg.get("argument"))
                normalized_events.append(normalized)
            docs.append({"doc_id": doc_id, "events": normalized_events, "arg_positions": {}})
    return docs


def iter_event_arguments(event):
    event_type = event.get("EventType")
    for role, value in event.items():
        if role == "EventType" or value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for one_value in values:
            if one_value is None:
                continue
            if isinstance(one_value, dict):
                text = one_value.get("text", one_value.get("argument", ""))
                start = one_value.get("start", one_value.get("arg_start"))
                end = one_value.get("end", one_value.get("arg_end"))
                yield event_type, role, text, start, end
            else:
                yield event_type, role, str(one_value), None, None


def arg_key_for_text(doc, text, start=None, end=None):
    if start is not None and end is not None:
        return (int(start), int(end), str(text))
    positions = doc.get("arg_positions", {}).get(str(text), [])
    if positions:
        return tuple(positions[0])
    return str(text)


def stringify_key(key):
    if isinstance(key, (tuple, list)):
        return json.dumps(list(key), ensure_ascii=False)
    return str(key)


def classify_doc_arguments(doc):
    arg_info = {}
    for event_index, event in enumerate(doc.get("events", [])):
        for event_type, role, text, start, end in iter_event_arguments(event):
            key = arg_key_for_text(doc, text, start, end)
            skey = stringify_key(key)
            if skey not in arg_info:
                arg_info[skey] = {
                    "arg_key": key,
                    "arg_text": str(text),
                    "event_types": set(),
                    "links": [],
                }
            arg_info[skey]["event_types"].add(event_type)
            arg_info[skey]["links"].append((event_index, event_type, role))

    for item in arg_info.values():
        if len(item["event_types"]) >= 2:
            item["category"] = "CROSS_TYPE_M2M"
        elif len(item["links"]) >= 2:
            item["category"] = "SAME_TYPE_MULTI"
        else:
            item["category"] = "SINGLE"
    return arg_info


def load_gold_categories(path):
    docs = normalize_docs(read_json(path))
    by_doc = {}
    for doc in docs:
        by_doc[doc["doc_id"]] = classify_doc_arguments(doc)
    return docs, by_doc


def extract_tuples(path):
    docs = normalize_docs(read_json(path))
    tuples = set()
    tuple_to_arg_key = {}
    for doc in docs:
        doc_id = doc["doc_id"]
        for event in doc.get("events", []):
            for event_type, role, text, start, end in iter_event_arguments(event):
                key = arg_key_for_text(doc, text, start, end)
                if isinstance(key, tuple):
                    tpl = (doc_id, event_type, role, key[0], key[1], key[2])
                else:
                    tpl = (doc_id, event_type, role, str(text))
                tuples.add(tpl)
                tuple_to_arg_key[tpl] = stringify_key(key)
    return tuples, tuple_to_arg_key
