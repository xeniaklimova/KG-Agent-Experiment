"""
enrich_nodedeg.py

Adds a `neighbor_info` text property to every node

Example property value on a Person node:
  "KNOWS_SN - Person: 3 | KNOWS_LW - Person: 1 | PARTY_TO - Crime: 1 | CURRENT_ADDRESS - Location: 1"

Usage:
    uv run python data_prep/enrich_nodedeg.py
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

DATABASE = "enrich3"

LABEL_RELATIONSHIPS: dict[str, list[tuple[str, str, str]]] = {
    "Person": [
        ("KNOWS_SN",        "Person",   "both"),
        ("KNOWS",           "Person",   "both"),
        ("KNOWS_LW",        "Person",   "both"),
        ("KNOWS_PHONE",     "Person",   "both"),
        ("FAMILY_REL",      "Person",   "both"),
        ("HAS_PHONE",       "Phone",    "out"),
        ("HAS_EMAIL",       "Email",    "out"),
        ("CURRENT_ADDRESS", "Location", "out"),
        ("PARTY_TO",        "Crime",    "out"),
    ],
    "Officer": [
        ("INVESTIGATED_BY", "Crime",    "in"),
    ],
    "Crime": [
        ("PARTY_TO",        "Person",   "in"),
        ("INVESTIGATED_BY", "Officer",  "out"),
        ("INVOLVED_IN",     "Vehicle",  "in"),
        ("OCCURRED_AT",     "Location", "out"),
    ],
    "Vehicle": [
        ("INVOLVED_IN",     "Crime",    "out"),
    ],
    "Location": [
        ("CURRENT_ADDRESS", "Person",   "in"),
        ("OCCURRED_AT",     "Crime",    "in"),
        ("LOCATION_IN_AREA","Area",     "out"),
        ("HAS_POSTCODE",    "PostCode", "out"),
    ],
    "Phone": [
        ("HAS_PHONE",       "Person",   "in"),
        ("CALLER",          "PhoneCall","out"),
        ("CALLED",          "PhoneCall","out"),
    ],
    "PhoneCall": [
        ("CALLER",          "Phone",    "in"),
        ("CALLED",          "Phone",    "in"),
    ],
    "Area": [
        ("LOCATION_IN_AREA","Location", "in"),
        ("POSTCODE_IN_AREA","PostCode", "in"),
    ],
    "PostCode": [
        ("HAS_POSTCODE",    "Location", "in"),
        ("POSTCODE_IN_AREA","Area",     "out"),
    ],
}


def _direction_pattern(rel_type: str, direction: str) -> str:
    if direction == "out":
        return f"(n)-[:`{rel_type}`]->()"
    elif direction == "in":
        return f"(n)<-[:`{rel_type}`]-()"
    else:
        return f"(n)-[:`{rel_type}`]-()"


def _build_query(label: str, rels: list[tuple[str, str, str]]) -> str:
    """
    Builds a Cypher query that sets neighbor_info on all nodes of `label`.
    Each entry in rels is (rel_type, neighbour_label, direction).
    Only non-zero counts are included in the summary string.
    """
    count_exprs = ", ".join(
        f"size([{_direction_pattern(r, d)} | 1]) AS c_{i}"
        for i, (r, _, d) in enumerate(rels)
    )

    parts = " + ".join(
        f"(CASE WHEN c_{i} > 0 "
        f"THEN '{r} - {nl}: ' + toString(c_{i}) + ' | ' "
        f"ELSE '' END)"
        for i, (r, nl, _) in enumerate(rels)
    )

    # Trim trailing separator
    summary_expr = f"substring({parts}, 0, size({parts}) - 3)"

    return f"""
        MATCH (n:{label})
        WITH n, {count_exprs}
        SET n.neighbor_info = CASE
            WHEN ({parts}) = '' THEN ''
            ELSE {summary_expr}
        END
        RETURN count(n) AS updated
    """


def enrich(driver: GraphDatabase.driver) -> None:
    with driver.session(database=DATABASE) as session:
        for label, rels in LABEL_RELATIONSHIPS.items():
            query = _build_query(label, rels)
            result = session.run(query)
            n = result.single()["updated"]
            print(f"  {label:12s}  {n:6d} nodes updated")


def verify(driver: GraphDatabase.driver) -> None:
    with driver.session(database=DATABASE) as session:
        print("\nVerification — sample neighbor_info values:")
        for label in LABEL_RELATIONSHIPS:
            row = session.run(
                f"MATCH (n:{label}) WHERE n.neighbor_info IS NOT NULL "
                f"AND n.neighbor_info <> '' "
                f"RETURN n.neighbor_info AS s LIMIT 1"
            ).single()
            sample = row["s"] if row else "none"
            print(f"  {label:12s}  {sample}")

        print("\nCoverage:")
        for label in LABEL_RELATIONSHIPS:
            row = session.run(
                f"MATCH (n:{label}) RETURN "
                f"count(n) AS total, "
                f"sum(CASE WHEN n.neighbor_info IS NOT NULL THEN 1 ELSE 0 END) AS enriched"
            ).single()
            print(f"  {label:12s}  {row['enriched']}/{row['total']}")


if __name__ == "__main__":
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    print(f"Adding neighbor_info to '{DATABASE}'...")
    enrich(driver)
    verify(driver)
    driver.close()
    print("Done.")
