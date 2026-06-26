"""
enrich_relprop.py

Adds `description` properties to all relationships in enrichrelprop.
Descriptions encode semantic meaning and selection hints for ambiguous
relationship types (KNOWS variants, CALLER/CALLED).

Usage:
    uv run python data_prep/enrich_relprop.py

"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

DATABASE = "enrichrelprop"

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


def enrich(driver: GraphDatabase.driver) -> None:
    with driver.session(database=DATABASE) as session:
        for rel_type, description in RELATIONSHIP_DESCRIPTIONS:
            result = session.run(
                f"MATCH ()-[r:`{rel_type}`]->() SET r.description = $desc RETURN count(r) AS n",
                desc=description,
            )
            n = result.single()["n"]
            print(f"  {rel_type:20s}  {n:6d} relationships  →  {description}")


def verify(driver: GraphDatabase.driver) -> None:
    with driver.session(database=DATABASE) as session:
        rows = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS total,
                   sum(CASE WHEN r.description IS NOT NULL THEN 1 ELSE 0 END) AS enriched
            ORDER BY rel_type
            """
        ).data()

    print("\nVerification:")
    unenriched = [r for r in rows if r["enriched"] < r["total"]]
    for row in rows:
        status = "OK" if row["enriched"] == row["total"] else "MISSING"
        print(f"  {row['rel_type']:20s}  {row['enriched']}/{row['total']}  {status}")

    if unenriched:
        print(f"\nWARNING: {len(unenriched)} relationship type(s) not fully enriched.")
    else:
        print("\nAll relationships enriched.")


if __name__ == "__main__":
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    print(f"Enriching relationships in '{DATABASE}'...")
    enrich(driver)
    verify(driver)
    driver.close()
    print("Done.")
