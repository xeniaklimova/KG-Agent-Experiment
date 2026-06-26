"""
prepare_ground_truth.py

Executes the gold Cypher query (mr) for every row in the filtered length
datasets and stores the results as answer_values.

For node results extracts identifiable properties as a list of dicts.
For scalar results stores raw values as a list.
For count queries stores the integer count as a single-element list.

Outputs:
    data/zograscope_length_train_v1_answered.csv
    data/zograscope_length_test_v1_answered.csv

New columns added:
    answer_values      — JSON list of results
    error_message      — empty string on success, error text on failure

Usage:
    uv run eval/prepare_ground_truth.py
"""

import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.graph import Node
from tqdm import tqdm

load_dotenv(dotenv_path=".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")

SPLITS = {
    "train": DATA_DIR / "zograscope_length_train_v1_filtered.csv",
    "test":  DATA_DIR / "zograscope_length_test_v1_filtered.csv",
}

IDENTIFIABLE_PROPS: dict[str, list[str]] = {
    "Person":    ["name", "surname", "nhs_no"],
    "Officer":   ["name", "surname", "badge_no", "rank"],
    "Crime":     ["id", "type", "date", "last_outcome"],
    "Vehicle":   ["reg", "make", "model", "year"],
    "Location":  ["address", "postcode"],
    "Object":    ["id", "type", "description"],
    "PhoneCall": ["call_type", "call_date", "call_time", "call_duration"],
    "Phone":     ["phoneNo"],
    "Email":     ["email_address"],
    "Area":      ["areaCode"],
    "PostCode":  ["code"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_node(node: Node) -> dict:
    """Extract identifiable properties from a Neo4j node."""
    labels = list(node.labels)
    label  = labels[0] if labels else None
    props  = IDENTIFIABLE_PROPS.get(label, [])
    result = {p: node.get(p) for p in props if node.get(p) is not None}
    if label in ("Person", "Officer"):
        name    = node.get("name", "")
        surname = node.get("surname", "")
        if name and surname:
            result["full_name"] = f"{name} {surname}"
    return result


def process_results(result) -> list:
    """
    Convert raw Neo4j result to a serialisable list.
    Iterates raw records (not .data()) to preserve Node types.
    """
    out = []
    for record in result:
        values = list(record.values())
        if len(values) == 1:
            v = values[0]
            if isinstance(v, Node):
                out.append(extract_node(v))
            else:
                out.append(v)
        else:
            row = {}
            for k, v in record.items():
                row[k] = extract_node(v) if isinstance(v, Node) else v
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    db = os.environ["NEO4J_DATABASE"]

    for split, src_path in SPLITS.items():
        dst_path = DATA_DIR / f"zograscope_length_{split}_v1_answered.csv"

        with open(src_path, newline="") as f:
            rows = list(csv.DictReader(f))
            fieldnames = list(rows[0].keys()) + [
                "answer_values",
                "execution_success",
                "error_message",
            ]

        results = []
        errors  = 0

        with driver.session(database=db) as session:
            for row in tqdm(rows, desc=split):
                cypher = row["mr"].strip()
                try:
                    answer  = process_results(session.run(cypher))
                    row["answer_values"]     = json.dumps(answer, default=str)
                    row["execution_success"] = "true"
                    row["error_message"]     = ""
                except Exception as e:
                    row["answer_values"]     = json.dumps([])
                    row["execution_success"] = "false"
                    row["error_message"]     = str(e)
                    errors += 1

                results.append(row)

        with open(dst_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"{split}: {len(rows)} rows → {errors} failures → {dst_path.name}")

    driver.close()


if __name__ == "__main__":
    main()
