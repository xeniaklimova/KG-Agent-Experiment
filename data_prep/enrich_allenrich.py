"""
enrich_allenrich.py

Applies all enrichments to the 'allenrich' database in one pass:
  1. Relationship descriptions (enrich_relprop)
  2. Shortcut edges            (enrich_edge)
  3. Neighbor-info properties  (enrich_nodedeg)

Usage:
    uv run python data_prep/enrich_allenrich.py
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

DATABASE = "allenrich"

# ── 1. Relationship descriptions ──────────────────────────────────────────────

RELATIONSHIP_DESCRIPTIONS: list[tuple[str, str]] = [
    # Contact / identity
    ("HAS_PHONE",        "has this phone number"),
    ("HAS_EMAIL",        "has this email address"),

    # Social — KNOWS variants are distinct by keyword
    ("KNOWS_SN",         "is friends (social network)"),
    ("KNOWS",            "is acquaintance of"),
    ("KNOWS_LW",         "lives with"),
    ("KNOWS_PHONE",      "knows phone number of"),
    ("FAMILY_REL",       "has family relations (rel_type gives parent/sibling/etc) with"),

    # Address / geography
    ("CURRENT_ADDRESS",  "person lives at this location"),
    ("OCCURRED_AT",      "crime occurred at this location"),
    ("LOCATION_IN_AREA", "location belongs to this area"),
    ("HAS_POSTCODE",     "location has this postcode"),
    ("POSTCODE_IN_AREA", "postcode belongs to this area"),

    # Crime
    ("PARTY_TO",         "person is party to this crime"),
    ("INVESTIGATED_BY",  "crime investigated by this officer"),
    ("INVOLVED_IN",      "person is involved in this crime"),

    # Phone calls — direction matters
    ("CALLER",           "call made from this phone"),
    ("CALLED",           "call received by this phone"),
]


def enrich_relprop(driver: GraphDatabase.driver) -> None:
    print(f"\n[1/3] Adding relationship descriptions to '{DATABASE}'...")
    with driver.session(database=DATABASE) as session:
        for rel_type, description in RELATIONSHIP_DESCRIPTIONS:
            result = session.run(
                f"MATCH ()-[r:`{rel_type}`]->() SET r.description = $desc RETURN count(r) AS n",
                desc=description,
            )
            n = result.single()["n"]
            print(f"  {rel_type:20s}  {n:6d} relationships  →  {description}")


# ── 2. Shortcut edges ─────────────────────────────────────────────────────────

SHORTCUT_EDGES: list[tuple[str, str, str]] = [
    # (description, cypher_match, merge_pattern)
    (
        "Person -[:LIVES_IN_AREA]-> Area  (via CURRENT_ADDRESS → LOCATION_IN_AREA)",
        "MATCH (p:Person)-[:CURRENT_ADDRESS]->(l:Location)-[:LOCATION_IN_AREA]->(a:Area)",
        "MERGE (p)-[:LIVES_IN_AREA]->(a)",
    ),
    (
        "Person -[:LIVES_IN_AREA]-> Area  (via CURRENT_ADDRESS → HAS_POSTCODE → POSTCODE_IN_AREA)",
        "MATCH (p:Person)-[:CURRENT_ADDRESS]->(l:Location)-[:HAS_POSTCODE]->(pc:PostCode)-[:POSTCODE_IN_AREA]->(a:Area)",
        "MERGE (p)-[:LIVES_IN_AREA]->(a)",
    ),
    (
        "Crime -[:OCCURRED_IN_AREA]-> Area  (via OCCURRED_AT → LOCATION_IN_AREA)",
        "MATCH (c:Crime)-[:OCCURRED_AT]->(l:Location)-[:LOCATION_IN_AREA]->(a:Area)",
        "MERGE (c)-[:OCCURRED_IN_AREA]->(a)",
    ),
    (
        "Crime -[:OCCURRED_IN_AREA]-> Area  (via OCCURRED_AT → HAS_POSTCODE → POSTCODE_IN_AREA)",
        "MATCH (c:Crime)-[:OCCURRED_AT]->(l:Location)-[:HAS_POSTCODE]->(pc:PostCode)-[:POSTCODE_IN_AREA]->(a:Area)",
        "MERGE (c)-[:OCCURRED_IN_AREA]->(a)",
    ),
    (
        "Person -[:CALLED_PERSON]-> Person  (via HAS_PHONE → PhoneCall)",
        "MATCH (p1:Person)-[:HAS_PHONE]->(:Phone)-[:CALLER|CALLED]-(pc:PhoneCall)-[:CALLER|CALLED]-(:Phone)<-[:HAS_PHONE]-(p2:Person) WHERE p1 <> p2",
        "MERGE (p1)-[:CALLED_PERSON]->(p2)",
    ),
    (
        "Person -[:IN_POSTCODE]-> PostCode  (via CURRENT_ADDRESS → HAS_POSTCODE)",
        "MATCH (p:Person)-[:CURRENT_ADDRESS]->(l:Location)-[:HAS_POSTCODE]->(pc:PostCode)",
        "MERGE (p)-[:IN_POSTCODE]->(pc)",
    ),
]


def enrich_edges(driver: GraphDatabase.driver) -> None:
    print(f"\n[2/3] Adding shortcut edges to '{DATABASE}'...")
    with driver.session(database=DATABASE) as session:
        for description, match_clause, merge_clause in SHORTCUT_EDGES:
            result = session.run(
                f"{match_clause} {merge_clause} RETURN count(*) AS n"
            )
            n = result.single()["n"]
            print(f"  {n:8d} edges  ←  {description}")


# ── 3. Neighbor-info node properties ─────────────────────────────────────────

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


def _build_neighbor_info_query(label: str, rels: list[tuple[str, str, str]]) -> str:
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


def enrich_nodedeg(driver: GraphDatabase.driver) -> None:
    print(f"\n[3/3] Adding neighbor_info properties to '{DATABASE}'...")
    with driver.session(database=DATABASE) as session:
        for label, rels in LABEL_RELATIONSHIPS.items():
            query = _build_neighbor_info_query(label, rels)
            result = session.run(query)
            n = result.single()["updated"]
            print(f"  {label:12s}  {n:6d} nodes updated")


# ── Verification ──────────────────────────────────────────────────────────────

def verify(driver: GraphDatabase.driver) -> None:
    print("\nVerification:")
    with driver.session(database=DATABASE) as session:

        # Relationship descriptions
        rows = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS total,
                   sum(CASE WHEN r.description IS NOT NULL THEN 1 ELSE 0 END) AS enriched
            ORDER BY rel_type
            """
        ).data()
        unenriched = [r for r in rows if r["enriched"] < r["total"]]
        if unenriched:
            print(f"  WARNING: {len(unenriched)} relationship type(s) missing descriptions: "
                  f"{[r['rel_type'] for r in unenriched]}")
        else:
            print(f"  Relationship descriptions: all {len(rows)} types enriched.")

        # Neighbor info
        missing = []
        for label in LABEL_RELATIONSHIPS:
            row = session.run(
                f"MATCH (n:{label}) RETURN "
                f"count(n) AS total, "
                f"sum(CASE WHEN n.neighbor_info IS NOT NULL THEN 1 ELSE 0 END) AS enriched"
            ).single()
            if row["enriched"] < row["total"]:
                missing.append(label)
        if missing:
            print(f"  WARNING: neighbor_info missing on labels: {missing}")
        else:
            print(f"  Neighbor info: all {len(LABEL_RELATIONSHIPS)} labels covered.")


if __name__ == "__main__":
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    enrich_relprop(driver)
    enrich_edges(driver)
    enrich_nodedeg(driver)
    verify(driver)
    driver.close()
    print("\nDone.")
