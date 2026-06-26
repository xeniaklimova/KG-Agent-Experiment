"""
enrich_edge.py

Adds shortcut edges to the enrichedge graph for faster agent traversal.

Usage:
    uv run python data_prep/enrich_relprop.py

"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

DATABASE = "enrichedge"

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


def add_shortcut_edges(driver: GraphDatabase.driver) -> None:
    with driver.session(database=DATABASE) as session:
        for description, match_clause, merge_clause in SHORTCUT_EDGES:
            result = session.run(
                f"{match_clause} {merge_clause} RETURN count(*) AS n"
            )
            n = result.single()["n"]
            print(f"  {n:8d} edges  ←  {description}")


if __name__ == "__main__":
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    print(f"Adding shortcut edges to '{DATABASE}'...")
    add_shortcut_edges(driver)
    driver.close()
    print("Done.")
