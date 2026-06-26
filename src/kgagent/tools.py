from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import tool

from kgagent.config import (
    NAME_PROPERTIES,
    ENTITY_LIST_LIMIT,
    TOOL_MSG_MAX_CHARS,
    DISCOVERY_RELATION_LIMIT,
)

# Module-level graph_db — injected by launcher scripts via set_graph_db()
graph_db = None

# Module-level flag — set to True by en1/en4 launcher scripts to enable relationship descriptions in _explore_discovery()
EXPLORE_WITH_DESCRIPTIONS: bool = False


def set_graph_db(db) -> None:
    """Inject the Neo4jGraph instance to use for all tool queries."""
    global graph_db
    graph_db = db


def _normalize_date(value: str) -> str:
    """Zero-pad day and month in D/M/YYYY date strings to match stored DD/MM/YYYY format."""
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', str(value).strip())
    if m:
        d, mo, y = m.groups()
        return f"{int(d):02d}/{int(mo):02d}/{y}"
    return value


@tool
def find_nodes(
    label:      str,
    text:       str = "",
    filters:    dict[str, str] = {},
    match_mode: str = "exact",
    limit:      int = ENTITY_LIST_LIMIT,
) -> str:
    """
    Find nodes by label. Use exactly ONE mode:
      text=...     Fuzzy name/description search (exact -> lower -> contains).
                   Use when you don't know the exact stored value.
      filters={..} Match on one or more properties.
                   match_mode: "exact" (default) | "contains" | "case_insensitive"

    Identifiers: Person->nhs_no, Officer->badge_no, Vehicle->reg, Location->postcode,
      Phone->phoneNo, Email->email_address, Area->areaCode, PostCode->code, Crime->id.

    Returns [{id, labels, props}, ...] or {truncated, count} if over limit.
    """
    if text:
        search_props = NAME_PROPERTIES.get(label, ["name"])

        def _or(template: str) -> str:
            return " OR ".join(template.format(p=f"n.`{p}`") for p in search_props)

        strategies = [
            ("exact",            _or("{p} = $text")),
            ("case_insensitive", _or("toLower({p}) = toLower($text)")),
            ("contains",         _or("toLower({p}) CONTAINS toLower($text)")),
        ]

        for match_type, where_clause in strategies:
            query = f"""
                MATCH (n:{label})
                WHERE {where_clause}
                RETURN elementId(n)  AS id,
                       labels(n)     AS labels,
                       properties(n) AS props
                LIMIT $limit
            """
            try:
                rows = graph_db.query(query, {"text": text, "limit": limit})
            except Exception as e:
                return json.dumps({"error": True, "message": str(e)})

            if rows:
                results = []
                for row in rows:
                    node_props = {
                        k: v for k, v in (row.get("props") or {}).items()
                        if k != "embedding"
                    }
                    matched_prop = next(
                        (p for p in search_props if node_props.get(p) is not None),
                        None,
                    )
                    name = next(
                        (str(node_props[p]) for p in search_props if node_props.get(p)),
                        None,
                    )
                    results.append({
                        "id":               row["id"],
                        "name":             name,
                        "labels":           row["labels"],
                        "props":            node_props,
                        "matched_property": matched_prop,
                        "match_type":       match_type,
                    })
                return json.dumps(results, default=str)

        return json.dumps([])

    if not filters:
        return json.dumps({
            "error":   True,
            "message": "Provide either text= (fuzzy search) or filters= (property match).",
        })

    conditions: list[str] = []
    params: dict[str, Any] = {"limit": limit}

    for prop, value in filters.items():
        param_name = f"val_{prop}"
        params[param_name] = _normalize_date(value)
        if match_mode == "exact":
            conditions.append(f"n.`{prop}` = ${param_name}")
        elif match_mode == "case_insensitive":
            conditions.append(f"toLower(n.`{prop}`) = toLower(${param_name})")
        elif match_mode == "contains":
            conditions.append(f"toLower(n.`{prop}`) CONTAINS toLower(${param_name})")
        else:
            return json.dumps({
                "error":   True,
                "message": f"Unknown match_mode '{match_mode}'. Use 'exact', 'contains', or 'case_insensitive'.",
            })

    where_clause = " AND ".join(conditions)

    try:
        rows = graph_db.query(
            f"""
            MATCH (n:{label})
            WHERE {where_clause}
            RETURN elementId(n)  AS id,
                   labels(n)     AS labels,
                   properties(n) AS props
            LIMIT $limit
            """,
            params,
        )
    except Exception as e:
        return json.dumps({"error": True, "message": str(e)})

    if not rows:
        return json.dumps([])

    results = [
        {
            "id":     row["id"],
            "labels": row["labels"],
            "props":  {k: v for k, v in (row.get("props") or {}).items()
                       if k != "embedding"},
        }
        for row in rows
    ]

    if len(results) >= limit:
        return json.dumps({
            "results":   results,
            "truncated": True,
            "count":     len(results),
            "message":   f"Result limit ({limit}) reached. Refine filters or increase limit.",
        }, default=str)

    return json.dumps(results, default=str)


@tool
def node_feature(node_ids: list[str], feature_name: str) -> str:
    """
    Fetch ONE property for one or more nodes. Pass ALL node_ids at once — never
    call once per node. Try explore(..., include_props=True) first; use this only
    for properties not visible there.
    Returns {"feature_name": ..., "values": {node_id: value, ...}}.
    """
    if not node_ids:
        return json.dumps({"error": True, "message": "node_ids must not be empty."})
    try:
        rows = graph_db.query(
            "UNWIND $ids AS nid MATCH (n) WHERE elementId(n) = nid "
            "RETURN nid AS node_id, n[$feat] AS value",
            {"ids": node_ids, "feat": feature_name},
        )
    except Exception as e:
        return json.dumps({"error": True, "message": str(e)})

    values = {r["node_id"]: r["value"] for r in rows}
    for nid in node_ids:
        values.setdefault(nid, None)
    return json.dumps({"feature_name": feature_name, "values": values}, default=str)


def _explore_discovery(node_ids: list[str], with_descriptions: bool = False) -> dict[str, list[dict]]:
    """Discover relationship types/counts for one or more nodes in ONE query.

    When with_descriptions=True (used by en1/en4 which have r.description on
    relationships), the query also fetches the first distinct description per
    relationship type and includes it in each returned entry.
    """
    if not node_ids:
        return {}

    if with_descriptions:
        query = """
            UNWIND $ids AS nid
            MATCH (n)-[r]-(m)
            WHERE elementId(n) = nid
            WITH nid, n, r,
                 CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS dir
            RETURN nid AS node_id, type(r) AS relationship, dir AS direction, count(r) AS count,
                   collect(DISTINCT r.description)[0] AS description
            ORDER BY nid, dir, count DESC
        """
        rows = graph_db.query(query, {"ids": node_ids})
        per_node: dict[str, list[dict]] = {nid: [] for nid in node_ids}
        for r in rows:
            entry = {
                "relationship": r["relationship"],
                "direction":    r["direction"],
                "count":        r["count"],
            }
            if r.get("description"):
                entry["description"] = r["description"]
            per_node[r["node_id"]].append(entry)
        return per_node
    else:
        query = """
            UNWIND $ids AS nid
            MATCH (n)-[r]-(m)
            WHERE elementId(n) = nid
            WITH nid, n, r,
                 CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS dir
            RETURN nid AS node_id, type(r) AS relationship, dir AS direction, count(r) AS count
            ORDER BY nid, dir, count DESC
        """
        rows = graph_db.query(query, {"ids": node_ids})
        per_node: dict[str, list[dict]] = {nid: [] for nid in node_ids}
        for r in rows:
            per_node[r["node_id"]].append({
                "relationship": r["relationship"],
                "direction":    r["direction"],
                "count":        r["count"],
            })
        return per_node


def _fetch_names(node_ids: list[str]) -> dict[str, str | None]:
    """Fetch a display name for each node_id in one query."""
    if not node_ids:
        return {}
    try:
        rows = graph_db.query(
            "UNWIND $ids AS nid MATCH (n) WHERE elementId(n) = nid "
            "RETURN nid AS node_id, "
            "coalesce(n.name, n.type, n.address, n.call_type, n.areaCode, "
            "n.code, n.phoneNo, n.email_address) AS name",
            {"ids": node_ids},
        )
    except Exception:
        return {nid: None for nid in node_ids}
    result = {r["node_id"]: r["name"] for r in rows}
    for nid in node_ids:
        result.setdefault(nid, None)
    return result


def _format_discovery(per_node: dict[str, list[dict]], names: dict[str, str | None] | None = None) -> str:
    """Group node_ids that share an identical relation profile to avoid repetition."""
    groups: dict[str, dict] = {}
    order: list[str] = []
    for nid, relations in per_node.items():
        ranked = sorted(relations, key=lambda r: r.get("count", 0), reverse=True)
        shown = ranked[:DISCOVERY_RELATION_LIMIT]
        omitted = len(ranked) - len(shown)
        key = json.dumps({"shown": shown, "omitted": omitted}, sort_keys=True)
        if key not in groups:
            groups[key] = {"relations": shown, "omitted": omitted, "applies_to": [], "names": []}
            order.append(key)
        groups[key]["applies_to"].append(nid)
        if names is not None:
            groups[key]["names"].append(names.get(nid))

    out = []
    for key in order:
        g = groups[key]
        entry: dict[str, Any] = {"applies_to": g["applies_to"]}
        if names is not None:
            entry["names"] = g["names"]
        entry["relations"] = g["relations"]
        if not g["relations"]:
            entry["message"] = "No relationships found."
        elif g["omitted"] > 0:
            entry["omitted_relationship_types"] = g["omitted"]
            entry["note"] = (
                f"{g['omitted']} additional lower-count relationship type(s) not shown. "
                f"Traverse the listed relationships first; if none look relevant, "
                f"narrow the candidate set before exploring further."
            )
        out.append(entry)
    return json.dumps({"discovery": out}, default=str)


def _explore_traversal(
    node_ids:      list[str],
    relationship:  str,
    direction:     str = "any",
    limit:         int = ENTITY_LIST_LIMIT,
    include_props: bool = False,
) -> str:
    """Traverse one relationship from one or more source nodes, merged + de-duplicated."""
    patterns = {
        "outgoing": "(n)-[r:`{rel}`]->(nb)",
        "incoming": "(n)<-[r:`{rel}`]-(nb)",
        "any":      "(n)-[r:`{rel}`]-(nb)",
    }
    pattern = patterns.get(direction)
    if pattern is None:
        return json.dumps({
            "error": True,
            "message": f"Invalid direction '{direction}'. Use 'outgoing', 'incoming', or 'any'.",
        })

    props_clause = ",\n               properties(nb) AS raw_props" if include_props else ""

    query = f"""
        UNWIND $ids AS nid
        MATCH {pattern.format(rel=relationship)}
        WHERE elementId(n) = nid
        RETURN elementId(nb)                                          AS id,
               coalesce(nb.name, nb.type, nb.address,
                        nb.call_type, nb.areaCode, nb.code,
                        nb.phoneNo, nb.email_address)                 AS name,
               labels(nb)                                             AS labels,
               type(r)                                                AS relationship,
               CASE WHEN startNode(r) = n THEN 'outgoing'
                    ELSE 'incoming' END                               AS direction,
               nid                                                    AS source_id{props_clause}
        LIMIT $raw_cap
    """

    try:
        results = graph_db.query(query, {
            "ids":     node_ids,
            "raw_cap": (limit + 1) * max(len(node_ids), 1) + 50,
        })
    except Exception as e:
        return json.dumps({"error": True, "message": str(e)})

    if not results:
        return json.dumps([])

    merged: dict[str, dict] = {}
    for r in results:
        nid = r["id"]
        if nid not in merged:
            item: dict[str, Any] = {
                "id":        nid,
                "name":      r["name"],
                "labels":    r["labels"],
                "direction": r["direction"],
                "from":      [],
            }
            if include_props and r.get("raw_props"):
                item["props"] = {k: v for k, v in r["raw_props"].items() if k != "embedding"}
            merged[nid] = item
        if r["source_id"] not in merged[nid]["from"]:
            merged[nid]["from"].append(r["source_id"])

    items_all = list(merged.values())

    if len(items_all) > limit:
        return json.dumps({
            "count":     len(items_all),
            "truncated": True,
            "message":   (
                f"More than {limit} distinct neighbors across {len(node_ids)} source node(s). "
                f"Use count_nodes for accurate count, or add filters."
            ),
        })

    items = []
    budget = TOOL_MSG_MAX_CHARS
    for item in items_all:
        budget -= len(json.dumps(item, default=str))
        if budget < 0:
            omitted = len(items_all) - len(items)
            items.append({"truncated": True, "omitted": omitted,
                          "message": f"{omitted} more neighbors omitted (response cap). Use count_nodes or narrow the traversal."})
            break
        items.append(item)
    return json.dumps(items, default=str)


@tool
def explore(
    node_ids:      list[str],
    relationship:  str = "",
    direction:     str = "any",
    limit:         int = ENTITY_LIST_LIMIT,
    include_props: bool = False,
) -> str:
    """
    Discover or traverse from one or more nodes.
    ALWAYS pass the WHOLE current candidate list in one call — never once per node.

    DISCOVERY (relationship=""):
      Returns relationship types/directions/counts for your candidate set.
      Call BEFORE each traversal hop.

    TRAVERSAL (relationship="REL_TYPE"):
      Follows that relationship from ALL node_ids, merged and deduped.
      direction: "outgoing" | "incoming" | "any" (default).
      include_props=True returns full node properties.
      Returns [{id, name, labels, direction, from}] or {truncated, count}.
    """
    if not node_ids:
        return json.dumps({"error": True, "message": "node_ids must not be empty."})

    if not relationship:
        try:
            per_node = _explore_discovery(node_ids, with_descriptions=EXPLORE_WITH_DESCRIPTIONS)
        except Exception as e:
            return json.dumps({"error": True, "message": str(e)})
        return _format_discovery(per_node)

    return _explore_traversal(node_ids, relationship, direction, limit, include_props)


#  FILTERING & VERIFICATION TOOLS

@tool
def filter_by_constraint(
    node_ids:        list[str],
    property_name:   str,
    operator:        str,
    value:           str = "",
    return_property: str = "",
) -> str:
    """
    Filter or rank nodes by a property.
    operator: "=" | ">" | ">=" | "<" | "<=" | "!="  -> returns matching nodes.
              "contains"                              -> substring match (case-sensitive).
              "argmax" | "argmin"                     -> returns {"answer": value}.

    Use "contains" for long text properties (last_outcome, type, description)
    rather than "=" — e.g. operator="contains", value="no suspect".

    RANKING: ALWAYS call this after collecting the full candidate set from
    traversal — never skip it and return empty entities. Set return_property
    when ranking by one field but returning another (e.g. rank by date, return type).
    """
    if operator in ("argmax", "argmin"):
        order    = "DESC" if operator == "argmax" else "ASC"
        ret_prop = return_property or property_name
        query = f"""
            MATCH (n)
            WHERE elementId(n) IN $ids AND n[$prop] IS NOT NULL
            WITH n,
                 CASE
                   WHEN toString(n[$prop]) =~ '^[0-9]{{2}}/[0-9]{{2}}/[0-9]{{4}}$'
                   THEN split(toString(n[$prop]), '/')[2] +
                        split(toString(n[$prop]), '/')[1] +
                        split(toString(n[$prop]), '/')[0]
                   ELSE toString(n[$prop])
                 END AS sort_key
            RETURN n[$ret_prop] AS answer_value
            ORDER BY sort_key {order}
            LIMIT 1
        """
        params: dict[str, Any] = {"ids": node_ids, "prop": property_name, "ret_prop": ret_prop}
        try:
            rows = graph_db.query(query, params)
        except Exception as e:
            return json.dumps({"error": True, "message": str(e)})
        answer = _normalize_date(str(rows[0]["answer_value"])) if rows else None
        return json.dumps({"answer": answer, "property": ret_prop}, default=str)

    elif operator in ("=", ">", ">=", "<", "<=", "!="):
        query = f"""
            MATCH (n)
            WHERE elementId(n) IN $ids
              AND (
                CASE
                  WHEN n[$prop] IS NOT NULL AND $val =~ '^-?[0-9]+(\\\\.[0-9]+)?$'
                    THEN toFloat(toString(n[$prop])) {operator} toFloat($val)
                  ELSE toString(n[$prop]) {operator} $val
                END
              )
            RETURN elementId(n) AS id,
                   labels(n)    AS labels,
                   n[$prop]     AS actual_value
        """
        params = {"ids": node_ids, "prop": property_name, "val": _normalize_date(value)}

    elif operator == "contains":
        query = """
            MATCH (n)
            WHERE elementId(n) IN $ids
              AND toLower(toString(n[$prop])) CONTAINS toLower($val)
            RETURN elementId(n) AS id,
                   labels(n)    AS labels,
                   n[$prop]     AS actual_value
        """
        params = {"ids": node_ids, "prop": property_name, "val": value}

    else:
        return json.dumps({
            "error": True,
            "message": f"Unknown operator '{operator}'. Use =, >, >=, <, <=, !=, contains, argmax, argmin.",
        })

    try:
        rows = graph_db.query(query, params)
    except Exception as e:
        return json.dumps({"error": True, "message": str(e)})

    results = [
        {"id": r["id"], "labels": r["labels"], property_name: r["actual_value"]}
        for r in rows
    ]
    return json.dumps(results, default=str)


#  COUNTING TOOLS

@tool
def count_nodes(
    via_relationships: list[str],
    target_label: str,
    from_node_id: str = "",
    from_label: str = "",
    from_filters: dict = {},
    filters: dict = {},
) -> str:
    """
    Count nodes reachable via a relationship chain (use when explore would truncate).
    Anchor: from_node_id (single node) OR from_label + from_filters (category).
    """
    if not via_relationships:
        return json.dumps({"error": True, "message": "via_relationships must not be empty."})
    if not from_node_id and not from_label:
        return json.dumps({"error": True, "message": "Provide either from_node_id or from_label."})

    hops = []
    for i, rel in enumerate(via_relationships):
        alias = f"n{i+1}" if i < len(via_relationships) - 1 else "target"
        hops.append(f"-[:`{rel}`]-({alias})")

    anchor_clause = f"(anchor:{from_label})" if from_label else "(anchor)"
    chain = anchor_clause + "".join(hops[:-1]) + hops[-1].rstrip(")") + f":{target_label})"

    params: dict[str, Any] = {}
    where_clauses = []

    if from_node_id:
        params["anchor_id"] = from_node_id
        where_clauses.append("elementId(anchor) = $anchor_id")

    for prop, value in (from_filters or {}).items():
        param_name = f"anchor_{prop}"
        where_clauses.append(f"anchor.`{prop}` = ${param_name}")
        params[param_name] = value

    for prop, value in (filters or {}).items():
        param_name = f"filter_{prop}"
        where_clauses.append(f"target.`{prop}` = ${param_name}")
        params[param_name] = value

    where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
        MATCH {chain}
        {where_str}
        RETURN count(DISTINCT target) AS count
    """

    try:
        results = graph_db.query(query, params)
    except Exception as e:
        return json.dumps({"error": True, "message": str(e)})

    count_val = results[0]["count"] if results else 0
    return json.dumps({
        "count":        count_val,
        "target_label": target_label,
        "filters":      filters or {},
    })


#  SET OPERATIONS

@tool
def set_ops(ids_a: list[str], ids_b: list[str], op: str) -> str:
    """
    Combine two node ID lists.
    op: "intersect" — nodes in BOTH A and B ("X who also Y" / "X and Y")
        "union"     — nodes in EITHER A or B ("X or Y")
    Returns {"result": [...], "count": N}.
    """
    set_a, set_b = set(ids_a), set(ids_b)
    if op == "intersect":
        result = list(set_a & set_b)
    elif op == "union":
        result = list(set_a | set_b)
    else:
        return json.dumps({"error": True, "message": f"Unknown op '{op}'. Use 'intersect' or 'union'."})
    return json.dumps({"result": result, "count": len(result)})


#  TERMINATION

@tool
def finish(answer: str, entities: list, answer_type: str) -> str:
    """
    Submit the final answer. ALWAYS call this last — never write plain text.
    answer_type: "entity_set" | "attribute" | "count" | "boolean" | "empty"
    entities: human-readable only, never raw node IDs.
      argmax/argmin/max/min -> ["<single verbatim value>"], nothing else.
      count <=20 -> list entity names.   count >20 -> ["N"] (number only).
    """
    structured = {
        "answer":      answer,
        "entities":    entities,
        "answer_type": answer_type,
    }
    return json.dumps(structured)


#  TOOL REGISTRY

TOOLS = [
    find_nodes,
    node_feature,
    explore,
    filter_by_constraint,
    count_nodes,
    set_ops,
    finish,
]
