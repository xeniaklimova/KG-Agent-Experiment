from __future__ import annotations

import json
import time
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from kgagent import tools as _tools_module
from kgagent.config import (
    CONTEXT_KEEP_HEAD,
    CONTEXT_KEEP_TAIL,
    HARD_FINISH_STEP,
    TOKEN_BUDGET,
    MAX_EXPLORE_CALLS,
    MAX_STEPS,
    TOKEN_HARD_CAP,
    TIME_HARD_CAP,
    PRESSURE_STEP,
    CHECKPOINT_STEP,
    STREAK_TOOL,
    STREAK_THRESHOLD,
    STREAK_EXCLUDE_THRESHOLD,
)
from kgagent.prompts import SYSTEM_PROMPT


class AgentState(TypedDict):
    messages:          Annotated[list, add_messages]
    structured_result: dict | None
    step_count:        int
    total_tokens:      int    # cumulative LLM token usage this question — drives TOKEN_BUDGET nudge and TOKEN_HARD_CAP stop
    node_cache:        dict   # maps "node_id:feature_name" -> raw value, for node_feature dedup
    relations_cache:   dict   # maps node_id -> list of relation dicts (explore discovery)
    traversal_cache:   dict   # maps "node_id|relationship|direction" -> True once traversed
    call_history:      list   # serialised (tool, args) keys — loop detection
    post_ranking:      bool   # True after argmax/argmin — blocks multi-entity finish()
    finished:          bool   # True once finish() has been successfully accepted — ends the graph
    explore_call_count:    int    # number of explore() calls made this question
    start_time:            float  # time.time() when the question started
    names_cache:           dict   # maps node_id -> display name string
    fbc_called:            bool   # True once filter_by_constraint has been successfully executed
    pre_finish_hint_given: bool   # True once the empty-after-traversal finish() hint has fired


def _trim_messages(messages: list) -> list:
    if len(messages) <= CONTEXT_KEEP_HEAD + CONTEXT_KEEP_TAIL:
        return messages

    head = list(messages[:CONTEXT_KEEP_HEAD])
    idx = CONTEXT_KEEP_HEAD

    if isinstance(head[-1], AIMessage) and getattr(head[-1], "tool_calls", None):
        pending = {tc["id"] for tc in head[-1].tool_calls}
        while idx < len(messages) and pending:
            msg = messages[idx]
            if isinstance(msg, ToolMessage) and msg.tool_call_id in pending:
                pending.discard(msg.tool_call_id)
                head.append(msg)
                idx += 1
            else:
                break

    tail = list(messages[-CONTEXT_KEEP_TAIL:])
    while tail and isinstance(tail[0], ToolMessage):
        tail = tail[1:]

    head_ids = {id(m) for m in head}
    tail = [m for m in tail if id(m) not in head_ids]

    return head + tail


# Backoff schedule (seconds) for HTTP 429 "rate limit exceeded" errors from the LLM endpoints
_RATE_LIMIT_WAITS = [10, 30, 60, 120, 180]


def _extract_token_usage(response: AIMessage) -> int:
    """Total tokens for a single LLM response, across provider response shapes."""
    meta  = response.response_metadata or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    total = usage.get("total_tokens", 0)
    if not total and getattr(response, "usage_metadata", None):
        um = response.usage_metadata
        total = um.get("total_tokens") or (um.get("input_tokens", 0) + um.get("output_tokens", 0))
    return total or 0


def _consecutive_tool_streak(messages: list) -> tuple[str | None, int]:
    """
    Look back over AIMessage tool-call turns (most recent first) and return
    (tool_name, streak_length) for the run of consecutive turns that each
    made exactly ONE tool call of the SAME tool. Stops at the first turn
    that calls a different tool (or multiple tools).
    """
    streak_tool: str | None = None
    streak = 0
    for m in reversed(messages):
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        names = {tc["name"] for tc in m.tool_calls}
        if len(names) != 1:
            break
        name = next(iter(names))
        if streak_tool is None:
            streak_tool = name
            streak = 1
        elif name == streak_tool:
            streak += 1
        else:
            break
    return streak_tool, streak


def build_app(llm, schema_annotations: str = "", owner_module=None):
    """
    Build and compile the LangGraph agent application.

    Parameters
    ----------
    llm : a LangChain chat model
    schema_annotations : str
        Extra guidance text appended to the schema SystemMessage.
        Use prompts.SCHEMA_ANNOTATIONS["base"|"en1"|"en2"|"en3"|"en4"].
    owner_module : module, optional
        The calling launcher script's module object (pass sys.modules[__name__]).
        When provided, should_continue reads TIME_HARD_CAP and MAX_STEPS from
        owner_module at call time — this allows the eval_runner to mutate these
        values on the launcher script and have the changes take effect immediately.

    Returns
    -------
    Compiled LangGraph app.
    """

    TOOLS = _tools_module.TOOLS
    llm_with_tools = llm.bind_tools(TOOLS)

    def _invoke_with_retry(messages):
        """Invoke the LLM with retry/backoff on 429 rate-limit errors."""
        last_exc = None
        for attempt, wait in enumerate([0] + _RATE_LIMIT_WAITS):
            if wait:
                print(f"[retry] rate limit — waiting {wait}s (attempt {attempt}/{len(_RATE_LIMIT_WAITS)})")
                time.sleep(wait)
            try:
                return llm_with_tools.invoke(messages)
            except Exception as e:
                msg = str(e)
                if "429" not in msg and "rate limit" not in msg.lower() and "rated_limited" not in msg.lower():
                    raise
                last_exc = e
        raise last_exc

    def agent_node(state: AgentState) -> dict:
        """Reasoning step: LLM decides which tool to call next."""
        step          = state.get("step_count", 0)
        tokens_so_far = state.get("total_tokens", 0)
        explore_calls = state.get("explore_call_count", 0)
        start_time = state.get("start_time") or time.time()
        messages = _trim_messages(state["messages"])
        extra = ""

        # Repetition streak — feeds the soft nudge (Tier B) and explore warning.
        streak_tool, streak_len = _consecutive_tool_streak(state["messages"])
        at_streak_limit = streak_tool == STREAK_TOOL and streak_len >= STREAK_EXCLUDE_THRESHOLD

        # Budget exhausted — inject a strong text nudge to call finish().
        force_finish = (
            step >= HARD_FINISH_STEP
            or tokens_so_far >= TOKEN_BUDGET
            or explore_calls >= MAX_EXPLORE_CALLS
        )

        if force_finish:
            if tokens_so_far >= TOKEN_BUDGET:
                reason = f"{tokens_so_far} tokens used (budget {TOKEN_BUDGET})"
            elif explore_calls >= MAX_EXPLORE_CALLS:
                reason = f"{explore_calls} explore() calls used (limit {MAX_EXPLORE_CALLS})"
            else:
                reason = f"{step} steps used (limit {HARD_FINISH_STEP})"
            extra += (
                f"\n\n[FORCED FINISH] Budget exhausted — {reason}. You MUST call "
                f"finish() right now with the best answer you can give from "
                f"everything explored so far. If nothing useful was found, call "
                f"finish(answer='Unable to determine', entities=[], answer_type='empty')."
            )
        else:
            # Force a plan restatement before the model has sunk too much budget
            if step == CHECKPOINT_STEP:
                extra += (
                    f"\n\n[CHECKPOINT] {step} tool calls made. Do you already have "
                    f"candidate nodes that contain the answer? If so, call "
                    f"filter_by_constraint (for ranking/argmax/argmin) or count_nodes "
                    f"(for counts) NOW rather than exploring further. If still traversing, "
                    f"state the remaining hop-by-hop path in one sentence and continue — "
                    f"but do NOT call find_nodes again."
                )

            # Same single tool called many times in a row with no progress
            if streak_tool == STREAK_TOOL and streak_len >= STREAK_THRESHOLD:
                extra += (
                    f"\n\n[REPETITION WARNING] {streak_len} consecutive explore() calls "
                    f"with no filter_by_constraint, count_nodes, set_ops, or finish() in "
                    f"between. If you have candidate nodes, use filter_by_constraint for "
                    f"ranking or count_nodes for counting — do NOT keep exploring. If truly "
                    f"stuck, try a DIFFERENT relationship or call finish() now."
                )
                if at_streak_limit:
                    extra += (
                        f"\n\n[EXPLORE LIMIT] {streak_len} explore() calls in a row with no "
                        f"progress. You MUST now choose node_feature, filter_by_constraint, "
                        f"count_nodes, set_ops, or finish() — do NOT call explore() again "
                        f"until you have processed the candidates you already have."
                    )

            # re-warning before the hard cutoff kicks in.
            if step >= PRESSURE_STEP:
                remaining = HARD_FINISH_STEP - step
                extra += (
                    f"\n\n[STEP BUDGET] {step} steps used. {remaining} step(s) left "
                    f"before you'll be REQUIRED to call finish() with whatever you "
                    f"have. If you have an answer, call finish() now — don't wait "
                    f"for the cutoff."
                )

        if extra:
            messages = [SystemMessage(content=messages[0].content + extra)] + list(messages[1:])

        response = _invoke_with_retry(messages)
        call_tokens = _extract_token_usage(response)

        return {
            "messages":     [response],
            "step_count":   step + 1,
            "total_tokens": tokens_so_far + call_tokens,
            "start_time":   start_time,
        }

    def tool_node_with_finish(state: AgentState) -> dict:
        """Execute tool calls. Capture structured result when finish() is called."""
        last_msg          = state["messages"][-1]
        structured_result = state.get("structured_result")
        node_cache        = dict(state.get("node_cache") or {})
        relations_cache   = dict(state.get("relations_cache") or {})
        traversal_cache   = dict(state.get("traversal_cache") or {})
        call_history      = list(state.get("call_history") or [])
        call_history_set  = set(call_history)
        post_ranking      = bool(state.get("post_ranking", False))
        finished          = bool(state.get("finished", False))
        explore_call_count = int(state.get("explore_call_count", 0))
        names_cache           = dict(state.get("names_cache") or {})
        fbc_called            = bool(state.get("fbc_called", False))
        pre_finish_hint_given = bool(state.get("pre_finish_hint_given", False))

        tool_map = {t.name: t for t in TOOLS}
        new_messages = []

        for tc in last_msg.tool_calls:
            # Loop detection: identical (tool, args) already executed this question
            if tc["name"] != "finish":
                call_key = f"{tc['name']}|{json.dumps(tc['args'], sort_keys=True)}"
                if call_key in call_history_set:
                    new_messages.append(ToolMessage(
                        content=json.dumps({
                            "warning": "LOOP DETECTED",
                            "message": (
                                f"You already called {tc['name']} with these exact arguments. "
                                f"The result is already in your context — do not repeat it. "
                                f"Call finish() now with what you have found, or "
                                f"finish(answer='Unable to determine', entities=[], answer_type='empty') "
                                f"if no answer was found."
                            ),
                        }),
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    ))
                    continue
                call_history.append(call_key)
                call_history_set.add(call_key)

            if tc["name"] == "finish":
                entities = tc["args"].get("entities", [])
                # Guard 1: ranking answer must be a single value
                if post_ranking and len(entities) > 1:
                    new_messages.append(ToolMessage(
                        content=json.dumps({
                            "error": True,
                            "message": (
                                f"Ranking answer must have exactly 1 entity, got {len(entities)}: {entities}. "
                                f"Re-read the question — what single value does it ask for? "
                                f"Call finish() again with only that value."
                            ),
                        }),
                        tool_call_id=tc["id"],
                        name="finish",
                    ))
                # Guard 2: empty answer after traversal without filter_by_constraint
                elif (
                    entities == []
                    and bool(traversal_cache)
                    and not fbc_called
                    and not pre_finish_hint_given
                ):
                    new_messages.append(ToolMessage(
                        content=json.dumps({
                            "warning": "EMPTY ANSWER BLOCKED",
                            "message": (
                                "You traversed the graph but are submitting empty entities. "
                                "For ranking questions (most/least/earliest/latest): take the "
                                "node IDs from your last traversal result and call "
                                "filter_by_constraint(node_ids=[...], property_name='...', "
                                "operator='argmax' or 'argmin'). "
                                "For set questions: check whether your traversal returned nodes "
                                "and report them. "
                                "Only call finish(entities=[]) again if you are certain nothing "
                                "exists in the graph."
                            ),
                        }),
                        tool_call_id=tc["id"],
                        name="finish",
                    ))
                    pre_finish_hint_given = True
                else:
                    try:
                        structured_result = json.loads(_tools_module.finish.invoke(tc["args"]))
                    except Exception:
                        structured_result = tc["args"]
                    new_messages.append(ToolMessage(
                        content=json.dumps(structured_result),
                        tool_call_id=tc["id"],
                        name="finish",
                    ))
                    finished = True
                post_ranking = False

            elif tc["name"] == "node_feature":
                node_ids = tc["args"].get("node_ids") or []
                feature  = tc["args"].get("feature_name", "")

                if not node_ids:
                    result = json.dumps({"error": True, "message": "node_ids must not be empty."})
                else:
                    uncached_ids = [nid for nid in node_ids if f"{nid}:{feature}" not in node_cache]
                    try:
                        if uncached_ids:
                            fresh = json.loads(_tools_module.node_feature.invoke({
                                "node_ids": uncached_ids,
                                "feature_name": feature,
                            }))
                            for nid, val in fresh.get("values", {}).items():
                                node_cache[f"{nid}:{feature}"] = val
                        values = {nid: node_cache.get(f"{nid}:{feature}") for nid in node_ids}
                        result = json.dumps({"feature_name": feature, "values": values}, default=str)
                    except Exception as e:
                        result = json.dumps({"error": True, "message": str(e)})

                new_messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tc["id"],
                    name="node_feature",
                ))

            elif tc["name"] == "explore":
                explore_call_count += 1
                node_ids     = tc["args"].get("node_ids") or []
                relationship = tc["args"].get("relationship", "")

                if not node_ids:
                    result = json.dumps({"error": True, "message": "node_ids must not be empty."})
                elif not relationship:
                    uncached_ids = [nid for nid in node_ids if nid not in relations_cache]
                    try:
                        new_per_node = _tools_module._explore_discovery(
                            uncached_ids,
                            with_descriptions=_tools_module.EXPLORE_WITH_DESCRIPTIONS,
                        ) if uncached_ids else {}
                        for nid, relations in new_per_node.items():
                            relations_cache[nid] = relations
                        # Fetch names for nodes not yet in the names cache
                        uncached_name_ids = [nid for nid in node_ids if nid not in names_cache]
                        if uncached_name_ids:
                            names_cache.update(_tools_module._fetch_names(uncached_name_ids))
                        per_node = {nid: relations_cache.get(nid, []) for nid in node_ids}
                        result = _tools_module._format_discovery(per_node, names=names_cache)
                    except Exception as e:
                        result = json.dumps({"error": True, "message": str(e)})
                else:
                    direction = tc["args"].get("direction", "any")
                    cache_keys = [f"{nid}|{relationship}|{direction}" for nid in node_ids]
                    already_done = all(k in traversal_cache for k in cache_keys)

                    try:
                        result = _tools_module.explore.invoke(tc["args"])
                    except Exception as e:
                        result = json.dumps({"error": True, "message": str(e)})

                    for k in cache_keys:
                        traversal_cache[k] = True

                    if already_done:
                        note = (
                            f"[REPEAT PATH] You already traversed '{relationship}' "
                            f"({direction}) from this exact set of nodes in a previous "
                            f"step — this is the same result as before. If this isn't "
                            f"progressing toward the answer, go back to DISCOVER "
                            f"(explore with no relationship) on a different candidate "
                            f"set, try a different relationship, or call finish() with "
                            f"your best answer.\n"
                        )
                        result = note + result

                new_messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tc["id"],
                    name="explore",
                ))

            else:
                tool_fn = tool_map.get(tc["name"])
                if tool_fn is None:
                    result = json.dumps({"error": True, "message": f"Unknown tool '{tc['name']}'"})
                else:
                    try:
                        result = tool_fn.invoke(tc["args"])
                    except Exception as e:
                        result = json.dumps({"error": True, "message": str(e)})

                if tc["name"] == "filter_by_constraint":
                    fbc_called = True
                    if tc["args"].get("operator") in ("argmax", "argmin"):
                        post_ranking = True

                new_messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tc["id"],
                    name=tc["name"],
                ))

        return {
            "messages":            new_messages,
            "structured_result":   structured_result,
            "node_cache":          node_cache,
            "relations_cache":     relations_cache,
            "traversal_cache":     traversal_cache,
            "call_history":        call_history,
            "post_ranking":        post_ranking,
            "finished":            finished,
            "explore_call_count":  explore_call_count,
            "names_cache":         names_cache,
            "fbc_called":          fbc_called,
            "pre_finish_hint_given": pre_finish_hint_given,
        }

    def should_end_after_tools(state: AgentState) -> str:
        """After executing tools, stop if finish() was successfully accepted."""
        if state.get("finished", False):
            return END
        return "agent"

    def should_continue(state: AgentState) -> str:
        # Read TIME_HARD_CAP and MAX_STEPS dynamically from owner_module if provided
        # so that eval_runner mutations (e.g. TIME_HARD_CAP = retry_time_cap) take effect.
        _max_steps      = getattr(owner_module, "MAX_STEPS",      MAX_STEPS)      if owner_module else MAX_STEPS
        _token_hard_cap = getattr(owner_module, "TOKEN_HARD_CAP", TOKEN_HARD_CAP) if owner_module else TOKEN_HARD_CAP
        _time_hard_cap  = getattr(owner_module, "TIME_HARD_CAP",  TIME_HARD_CAP)  if owner_module else TIME_HARD_CAP

        # Unconditional hard stops — fire regardless
        if state.get("step_count", 0) >= _max_steps:
            return END
        if state.get("total_tokens", 0) >= _token_hard_cap:
            return END
        if time.time() - state.get("start_time", time.time()) >= _time_hard_cap:
            return END
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            if any(tc["name"] == "finish" for tc in last.tool_calls):
                return "tools_then_end"
            return "tools"
        return END

    def schema_node(state: AgentState) -> dict:
        """Inject node labels, properties, and valid relationship types."""
        graph_db = _tools_module.graph_db
        graph_db.refresh_schema()

        node_query = """
            CALL db.schema.nodeTypeProperties()
            YIELD nodeLabels, propertyName
            RETURN nodeLabels, collect(propertyName) AS properties
            ORDER BY nodeLabels
        """
        try:
            rows = graph_db.query(node_query)
            lines = []
            for r in rows:
                label = ", ".join(r["nodeLabels"]) if isinstance(r["nodeLabels"], list) else str(r["nodeLabels"])
                props = ", ".join(p for p in r["properties"] if p != "embedding")
                lines.append(f"  {label}: {props}")
            schema_text = "\n".join(lines)
        except Exception:
            schema_text = graph_db.schema

        # en1 and en4 use EXPLORE_WITH_DESCRIPTIONS — fetch descriptions inline
        if _tools_module.EXPLORE_WITH_DESCRIPTIONS:
            try:
                rel_rows = graph_db.query(
                    "MATCH ()-[r]-() "
                    "WITH type(r) AS relType, collect(DISTINCT r.description)[0] AS desc "
                    "RETURN relType, desc "
                    "ORDER BY relType"
                )
                rel_lines = []
                for r in rel_rows:
                    line = f"  {r['relType']}"
                    if r.get("desc"):
                        line += f" — {r['desc']}"
                    rel_lines.append(line)
                rel_types_text = "\n".join(rel_lines)
            except Exception:
                try:
                    rel_rows = graph_db.query(
                        "CALL db.relationshipTypes() YIELD relationshipType "
                        "RETURN relationshipType ORDER BY relationshipType"
                    )
                    rel_types_text = "  " + ", ".join(r["relationshipType"] for r in rel_rows)
                except Exception:
                    rel_types_text = "  unknown"
        else:
            try:
                rel_rows = graph_db.query(
                    "CALL db.relationshipTypes() YIELD relationshipType "
                    "RETURN relationshipType ORDER BY relationshipType"
                )
                rel_types = ", ".join(r["relationshipType"] for r in rel_rows)
                rel_types_text = f"  {rel_types}"
            except Exception:
                rel_types_text = "  unknown"

        schema_msg = SystemMessage(
            content=(
                f"NODE LABELS AND PROPERTIES:\n{schema_text}\n\n"
                f"VALID RELATIONSHIP TYPES (use ONLY these exact strings with explore's "
                f"traversal mode — never invent or guess names):\n{rel_types_text}\n\n"
                f"Use explore(node_ids=[...]) with no relationship to confirm which of these "
                f"exist on your candidate nodes and in which direction before traversing.\n\n"
                f"{schema_annotations}"
            ),
        )
        return {"messages": [schema_msg]}

    workflow = StateGraph(AgentState)
    workflow.add_node("schema", schema_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node_with_finish)

    workflow.add_edge(START, "schema")
    workflow.add_edge("schema", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools":          "tools",
            "tools_then_end": "tools",
            END:              END,
        },
    )
    workflow.add_conditional_edges(
        "tools",
        should_end_after_tools,
        {
            "agent": "agent",
            END:     END,
        },
    )

    return workflow.compile()


def ask(question: str, app, verbose: bool = True) -> dict:
    """

    Parameters
    ----------
    question : str
    app : compiled LangGraph app returned by build_app()
    verbose : bool

    Returns
    -------
    dict with keys: answer, entities, answer_type, terminated_early
    """
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
        "structured_result": None,
        "step_count":        0,
        "total_tokens":      0,
        "node_cache":        {},
        "relations_cache":   {},
        "traversal_cache":   {},
        "call_history":      [],
        "post_ranking":        False,
        "finished":            False,
        "explore_call_count":  0,
        "start_time":          time.time(),
        "names_cache":         {},
        "fbc_called":          False,
        "pre_finish_hint_given": False,
    }

    final_state = None
    for step in app.stream(initial_state, stream_mode="values"):
        final_state = step
        if verbose:
            last = step["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                for tc in last.tool_calls:
                    print(f"  -> {tc['name']}({tc['args']})")
            elif isinstance(last, ToolMessage):
                preview = last.content[:120].replace("\n", " ")
                print(f"     {preview}")

    result           = final_state.get("structured_result") or {}
    steps            = final_state.get("step_count", 0) if final_state else 0
    terminated_early = steps >= MAX_STEPS

    return {
        "answer":          result.get("answer", ""),
        "entities":        result.get("entities", []),
        "answer_type":     result.get("answer_type", "unknown"),
        "terminated_early": terminated_early,
    }
