import argparse
from pathlib import Path

from utils import classify_doc_arguments, normalize_docs, read_json, safe_div, write_json


def main():
    parser = argparse.ArgumentParser(description="Experiment 0 data statistics for multi-target arguments.")
    parser.add_argument("--gold_data", required=True)
    parser.add_argument("--output", default="outputs/exp0/data_stats.json")
    args = parser.parse_args()

    docs = normalize_docs(read_json(args.gold_data))
    total_args = 0
    cross_type_args = 0
    same_type_args = 0
    m2m_docs = 0
    per_doc = []

    for doc in docs:
        arg_info = classify_doc_arguments(doc)
        num_cross = sum(1 for x in arg_info.values() if x["category"] == "CROSS_TYPE_M2M")
        num_same = sum(1 for x in arg_info.values() if x["category"] == "SAME_TYPE_MULTI")
        event_types = {event.get("EventType") for event in doc.get("events", [])}
        total_args += len(arg_info)
        cross_type_args += num_cross
        same_type_args += num_same
        m2m_docs += int(len(doc.get("events", [])) >= 2 and len(event_types) >= 2)
        per_doc.append(
            {
                "doc_id": doc["doc_id"],
                "num_events": len(doc.get("events", [])),
                "num_event_types": len(event_types),
                "num_argument_spans": len(arg_info),
                "num_cross_type_multi_target_args": num_cross,
            }
        )

    output = {
        "total_documents": len(docs),
        "m2m_documents": m2m_docs,
        "m2m_document_ratio": safe_div(m2m_docs, len(docs)),
        "total_argument_spans": total_args,
        "cross_type_multi_target_arguments": cross_type_args,
        "cross_type_multi_target_argument_ratio": safe_div(cross_type_args, total_args),
        "same_type_multi_instance_arguments": same_type_args,
        "same_type_multi_instance_argument_ratio": safe_div(same_type_args, total_args),
        "documents": per_doc,
    }
    write_json(Path(args.output), output)


if __name__ == "__main__":
    main()
