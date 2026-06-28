"""

Runs the agent against a random sample of benchmark questions and
scores results against ground truth.

Usage:
    uv run eval/eval_runner.py [--n 50] [--split test] [--seed 42] [--agent agent3] [--out results/]

Output:
    results/eval_<agent>_<split>_n<N>_seed<S>_<timestamp>.csv   — per-question rows
    results/eval_<agent>_<split>_n<N>_seed<S>_<timestamp>.json  — summary statistics
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "scripts")

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage, HumanMessage


agent_module = None
app          = None
AgentState   = None
SYSTEM_PROMPT = None

csv.field_size_limit(sys.maxsize)

# EXCLUDED_TYPES = {"argmax", "argmin", "max", "min"}
EXCLUDED_TYPES = set()


PRICE_INPUT_PER_M  = 2.00
PRICE_OUTPUT_PER_M = 8.00
EUR_PER_USD        = 0.92




def run_agent(question: str) -> dict:
    """Run the agent on a single question. Returns structured result + usage stats."""
    if hasattr(agent_module, "get_initial_state"):
        initial_state = agent_module.get_initial_state(question)
    else:
        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
            "structured_result": None,
            "step_count":        0,
            "node_cache":        {},
        }

    llm_calls     = 0
    input_tokens  = 0
    output_tokens = 0
    tool_sequence = []
    final_state   = None

    stream_error = None
    try:
        for step in app.stream(initial_state, stream_mode="values"):
            msgs        = step.get("messages") or []
            last_msg    = msgs[-1] if msgs else None
            final_state = step

            if isinstance(last_msg, AIMessage):
                llm_calls += 1
                meta  = last_msg.response_metadata or {}
                usage = meta.get("token_usage") or meta.get("usage") or {}
                call_in  = usage.get("prompt_tokens",     0)
                call_out = usage.get("completion_tokens", 0)
                if call_in == 0 and hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
                    um = last_msg.usage_metadata
                    call_in  = um.get("input_tokens",  0)
                    call_out = um.get("output_tokens", 0)
                input_tokens  += call_in
                output_tokens += call_out
                if last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        tool_sequence.append(tc["name"])
    except Exception as e:
        stream_error = e

    result = {}
    if final_state and final_state.get("structured_result"):
        result = final_state["structured_result"]

    steps            = final_state.get("step_count", 0) if final_state else 0
    terminated_early = steps >= agent_module.MAX_STEPS
    cost = (input_tokens * PRICE_INPUT_PER_M + output_tokens * PRICE_OUTPUT_PER_M) / 1_000_000

    if stream_error is not None:
        raise RuntimeError(f"{stream_error} [at step {steps}]") from stream_error

    return {
        "answer":          result.get("answer", ""),
        "entities":        result.get("entities", []),
        "answer_type":     result.get("answer_type", "unknown"),
        "finish_called":   bool(result),
        "llm_calls":       llm_calls,
        "input_tokens":    input_tokens,
        "output_tokens":   output_tokens,
        "total_tokens":    input_tokens + output_tokens,
        "cost_usd":        cost,
        "step_count":      steps,
        "terminated_early": terminated_early,
        "tool_sequence":   tool_sequence,
    }


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    """Lowercase, strip whitespace, collapse internal spaces."""
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _dedup(entities: list) -> list:
    """Remove duplicate entries from a GT list, preserving order."""
    seen, out = set(), []
    for e in entities:
        key = json.dumps(e, sort_keys=True) if isinstance(e, dict) else str(e)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def _entity_strings(entity: dict | str) -> list[str]:
    """All normalised string values from a ground truth entity dict (or scalar)."""
    if isinstance(entity, dict):
        return [_normalise(v) for v in entity.values() if v is not None]
    return [_normalise(entity)]


def _match_entity(gt_entity: dict | str, predicted: list[str]) -> bool:
    """True if any predicted string matches any identifier string of gt_entity."""
    pred_norm = [_normalise(p) for p in predicted]
    for gt_str in _entity_strings(gt_entity):
        if not gt_str:
            continue
        for p in pred_norm:
            if gt_str == p or gt_str in p or p in gt_str:
                return True
    return False


def score_answer(row: dict, agent_result: dict) -> dict:
    """
    Compare agent answer against ground truth.

    Returns dict with:
        exact_match  — bool: predicted set == ground truth set (strict)
        f1           — float: entity-level F1 (lenient)
        count_correct — bool | None: for count questions, numeric match
        score_note   — str: brief explanation
    """
    q_type       = row["type"]
    raw_gt       = row.get("answer_values", "[]")
    predicted    = agent_result.get("entities", [])

    try:
        gt_parsed = json.loads(raw_gt)
    except (json.JSONDecodeError, TypeError):
        gt_parsed = []

    if q_type == "count":
        # Enriched: {"count": N, "entities": [...]}
        if isinstance(gt_parsed, dict) and "count" in gt_parsed:
            true_count  = gt_parsed["count"]
            gt_entities = _dedup(gt_parsed.get("entities", []))
        else:
            # Large count: [N]
            true_count  = gt_parsed[0] if gt_parsed else None
            gt_entities = []

        # Split predicted into count strings (integers) and entity names.
        # The agent may return a number ("3"), entity names, or a mix.
        count_preds  = []
        entity_preds = []
        for p in predicted:
            try:
                count_preds.append(int(str(p).strip()))
            except ValueError:
                entity_preds.append(p)

        # Determine agent's count answer
        if count_preds:
            agent_count = count_preds[0]
        elif entity_preds and gt_entities:
            # Agent returned entity names for a small count — infer count from list length
            agent_count = len(entity_preds)
        else:
            agent_count = None

        count_correct = (agent_count is not None and agent_count == true_count)


        if gt_entities and entity_preds:
            tp        = sum(1 for gt_e in gt_entities if _match_entity(gt_e, entity_preds))
            precision = tp / len(entity_preds)
            recall    = tp / len(gt_entities)
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            f1        = min(1.0, f1)
            exact     = count_correct and (tp == len(gt_entities) == len(entity_preds))
            note      = f"count: true={true_count} pred={agent_count} entity_f1={f1:.2f}"
        else:
            # Agent returned a count number only — score on count match alone
            f1    = 1.0 if count_correct else 0.0
            exact = bool(count_correct)
            note  = f"count only: true={true_count} pred={agent_count}"

        return {
            "exact_match":   exact,
            "f1":            round(f1, 4),
            "count_correct": count_correct,
            "score_note":    note,
        }

    gt_entities = _dedup(gt_parsed) if isinstance(gt_parsed, list) else []

    if not gt_entities:
        # Ground truth is empty — correct only if agent also says empty
        agent_empty = (not predicted or agent_result.get("answer_type") == "empty")
        return {
            "exact_match":   agent_empty,
            "f1":            1.0 if agent_empty else 0.0,
            "count_correct": None,
            "score_note":    "empty ground truth",
        }

    tp = sum(1 for gt_e in gt_entities if _match_entity(gt_e, predicted))
    precision = tp / len(predicted) if predicted else 0.0
    recall    = tp / len(gt_entities)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    f1 = min(1.0, f1)
    exact = (tp == len(gt_entities) and len(predicted) == len(gt_entities))

    return {
        "exact_match":   exact,
        "f1":            round(f1, 4),
        "count_correct": None,
        "score_note":    f"tp={tp} pred={len(predicted)} true={len(gt_entities)}",
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int,  default=50,    help="Number of questions to sample")
    parser.add_argument("--split", type=str,  default="test", choices=["train", "test", "mixed"])
    parser.add_argument("--seed",  type=int,  default=42)
    parser.add_argument("--agent", type=str,  default="base", choices=[ "en1", "base", "en2", "en3", "en4"],
                        help="Agent module to use (default: base)")
    parser.add_argument("--out",   type=str,  default="results", help="Output directory")
    args = parser.parse_args()

    # Dynamic import so the correct agent connects to the right database.
    import importlib
    global agent_module, app, AgentState, SYSTEM_PROMPT
    agent_module  = importlib.import_module(args.agent)
    app           = agent_module.app
    AgentState    = agent_module.AgentState
    SYSTEM_PROMPT = agent_module.SYSTEM_PROMPT

    print(f"Agent: {args.agent}")

    random.seed(args.seed)

    if args.split == "mixed":
        def _load(split):
            with open(f"data/zograscope_length_{split}_v1_answered_v3.csv") as f:
                rows = list(csv.DictReader(f))
            return [r for r in rows if r["type"] not in EXCLUDED_TYPES]

        train_rows = _load("train")
        test_rows  = _load("test")
        n_each     = args.n // 2
        sample     = (random.sample(train_rows, min(n_each, len(train_rows))) +
                      random.sample(test_rows,  min(n_each, len(test_rows))))
        random.shuffle(sample)
        print(f"Loaded {len(train_rows)} train + {len(test_rows)} test eligible questions")
        print(f"Sampled {len(sample)} questions ({n_each} easy + {n_each} hard, seed={args.seed})\n")
    else:
        with open(f"data/zograscope_length_{args.split}_v1_answered_v3.csv") as f:
            rows = list(csv.DictReader(f))
        rows   = [r for r in rows if r["type"] not in EXCLUDED_TYPES]
        sample = random.sample(rows, min(args.n, len(rows)))
        print(f"Loaded {len(rows)} eligible questions from {args.split} split")
        print(f"Sampled {len(sample)} questions (seed={args.seed})\n")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem      = f"eval_{args.agent}_{args.split}_n{len(sample)}_seed{args.seed}_{timestamp}"
    csv_out   = out_dir / f"{stem}.csv"
    json_out  = out_dir / f"{stem}.json"

    fieldnames = [
        "question", "type", "num_nodes",
        "exact_match", "f1", "count_correct", "score_note",
        "answer_type_pred", "answer_type_true",
        "finish_called", "terminated_early",
        "llm_calls", "step_count",
        "input_tokens", "output_tokens", "total_tokens", "cost_usd",
        "elapsed_s",
        "predicted_entities", "tool_sequence",
        "agent_answer",
    ]

    records = []
    retry_queue = []  # (record_index, benchmark_row) for errored questions

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(sample):
            q      = row["nl"]
            q_type = row["type"]
            n_nodes = row["num_nodes"]

            print(f"[{i+1}/{len(sample)}] type={q_type}  num_nodes={n_nodes}")
            print(f"  Q: {q}")

            t0 = time.time()
            try:
                agent_result = run_agent(q)
                elapsed = round(time.time() - t0, 1)
            except Exception as e:
                elapsed = round(time.time() - t0, 1)
                import re as _re
                _m = _re.search(r'\[at step (\d+)\]', str(e))
                err_steps = int(_m.group(1)) if _m else 0
                print(f"  ERROR (at step {err_steps}): {e} — queued for retry\n")
                rec = {
                    "question": q, "type": q_type, "num_nodes": n_nodes,
                    "exact_match": False, "f1": 0.0, "count_correct": None,
                    "score_note": f"error: {e}",
                    "answer_type_pred": "error", "answer_type_true": q_type,
                    "finish_called": False, "terminated_early": False,
                    "llm_calls": 0, "step_count": err_steps,
                    "input_tokens": 0, "output_tokens": 0,
                    "total_tokens": 0, "cost_usd": 0.0,
                    "elapsed_s": elapsed,
                    "predicted_entities": "[]",
                    "tool_sequence": "[]",
                    "agent_answer": "",
                }
                retry_queue.append((len(records), row))
                records.append(rec)
                writer.writerow(rec)
                f.flush()
                continue

            scores = score_answer(row, agent_result)

            print(f"  exact={scores['exact_match']}  f1={scores['f1']:.2f}  "
                  f"finish={agent_result['finish_called']}  "
                  f"{'⚠ TERMINATED' if agent_result['terminated_early'] else ''}")
            print(f"  predicted: {agent_result['entities']}")
            print(f"  note: {scores['score_note']}")
            print()

            rec = {
                "question":          q,
                "type":              q_type,
                "num_nodes":         n_nodes,
                "exact_match":       scores["exact_match"],
                "f1":                scores["f1"],
                "count_correct":     scores["count_correct"],
                "score_note":        scores["score_note"],
                "answer_type_pred":  agent_result["answer_type"],
                "answer_type_true":  q_type,
                "finish_called":     agent_result["finish_called"],
                "terminated_early":  agent_result["terminated_early"],
                "llm_calls":         agent_result["llm_calls"],
                "step_count":        agent_result["step_count"],
                "input_tokens":      agent_result["input_tokens"],
                "output_tokens":     agent_result["output_tokens"],
                "total_tokens":      agent_result["total_tokens"],
                "cost_usd":          round(agent_result["cost_usd"], 6),
                "elapsed_s":         elapsed,
                "predicted_entities": json.dumps(agent_result["entities"]),
                "tool_sequence":     json.dumps(agent_result["tool_sequence"]),
                "agent_answer":      agent_result.get("answer", ""),
            }
            records.append(rec)
            writer.writerow(rec)
            f.flush()  # write progressively so partial results are not lost

    # ── Retry errored questions with a shorter time cap ───────────────────────
    if retry_queue:
        original_time_cap = agent_module.TIME_HARD_CAP
        retry_time_cap    = 30  # shorter run = smaller context = fewer malformed calls
        agent_module.TIME_HARD_CAP = retry_time_cap
        print(f"\n{'='*60}")
        print(f"RETRY PASS: {len(retry_queue)} errored question(s) — TIME_HARD_CAP={retry_time_cap}s")
        print(f"{'='*60}\n")

        for rec_idx, row in retry_queue:
            q      = row["nl"]
            q_type = row["type"]
            n_nodes = row["num_nodes"]
            print(f"[retry] type={q_type}  num_nodes={n_nodes}")
            print(f"  Q: {q}")

            t0 = time.time()
            try:
                agent_result = run_agent(q)
                elapsed = round(time.time() - t0, 1)
            except Exception as e:
                elapsed = round(time.time() - t0, 1)
                print(f"  RETRY FAILED: {e}\n")
                records[rec_idx]["score_note"] += f" | retry_failed: {e}"
                continue

            scores = score_answer(row, agent_result)
            print(f"  exact={scores['exact_match']}  f1={scores['f1']:.2f}  "
                  f"finish={agent_result['finish_called']}")
            print(f"  predicted: {agent_result['entities']}")
            print(f"  note: {scores['score_note']}\n")

            records[rec_idx] = {
                "question":          q,
                "type":              q_type,
                "num_nodes":         n_nodes,
                "exact_match":       scores["exact_match"],
                "f1":                scores["f1"],
                "count_correct":     scores["count_correct"],
                "score_note":        scores["score_note"] + " [retry]",
                "answer_type_pred":  agent_result["answer_type"],
                "answer_type_true":  q_type,
                "finish_called":     agent_result["finish_called"],
                "terminated_early":  agent_result["terminated_early"],
                "llm_calls":         agent_result["llm_calls"],
                "step_count":        agent_result["step_count"],
                "input_tokens":      agent_result["input_tokens"],
                "output_tokens":     agent_result["output_tokens"],
                "total_tokens":      agent_result["total_tokens"],
                "cost_usd":          round(agent_result["cost_usd"], 6),
                "elapsed_s":         elapsed,
                "predicted_entities": json.dumps(agent_result["entities"]),
                "tool_sequence":     json.dumps(agent_result["tool_sequence"]),
                "agent_answer":      agent_result.get("answer", ""),
            }

        agent_module.TIME_HARD_CAP = original_time_cap

        # Rewrite CSV with retry results merged in
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(records)
    exact_acc   = sum(r["exact_match"] for r in records) / n
    avg_f1      = sum(r["f1"]          for r in records) / n
    finish_rate = sum(r["finish_called"] for r in records) / n
    term_rate   = sum(r["terminated_early"] for r in records) / n
    avg_tokens  = sum(r["total_tokens"] for r in records) / n
    avg_cost    = sum(r["cost_usd"]     for r in records) / n
    avg_steps   = sum(r["step_count"]   for r in records) / n
    total_cost  = sum(r["cost_usd"]     for r in records)
    finished    = [r for r in records if r["finish_called"]]
    avg_tokens_finished = (sum(r["total_tokens"] for r in finished) / len(finished)
                           if finished else 0)

    # Per-type breakdown
    from collections import defaultdict
    _stat_defaults = lambda: {
        "n": 0, "exact": 0, "f1_sum": 0.0,
        "steps_sum": 0, "tokens_sum": 0, "cost_sum": 0.0, "terminated": 0,
    }
    type_stats: dict[str, dict] = defaultdict(_stat_defaults)
    for r in records:
        t = r["type"]
        type_stats[t]["n"]          += 1
        type_stats[t]["exact"]      += int(r["exact_match"])
        type_stats[t]["f1_sum"]     += r["f1"]
        type_stats[t]["steps_sum"]  += r["step_count"]
        type_stats[t]["tokens_sum"] += r["total_tokens"]
        type_stats[t]["cost_sum"]   += r["cost_usd"]
        type_stats[t]["terminated"] += int(r["terminated_early"])

    # Per-complexity breakdown (num_nodes)
    complexity_stats: dict[str, dict] = defaultdict(_stat_defaults)
    for r in records:
        bucket = str(r.get("num_nodes", "?"))
        complexity_stats[bucket]["n"]          += 1
        complexity_stats[bucket]["exact"]      += int(r["exact_match"])
        complexity_stats[bucket]["f1_sum"]     += r["f1"]
        complexity_stats[bucket]["steps_sum"]  += r["step_count"]
        complexity_stats[bucket]["tokens_sum"] += r["total_tokens"]
        complexity_stats[bucket]["cost_sum"]   += r["cost_usd"]
        complexity_stats[bucket]["terminated"] += int(r["terminated_early"])

    easy_nodes    = {str(k): v for k, v in complexity_stats.items() if str(k) in ("2", "3")}
    complex_nodes = {str(k): v for k, v in complexity_stats.items() if str(k) in ("4", "5")}

    def _agg(group: dict) -> dict:
        n     = sum(v["n"]     for v in group.values())
        exact = sum(v["exact"] for v in group.values())
        f1s   = sum(v["f1_sum"] for v in group.values())
        steps  = sum(v["steps_sum"]  for v in group.values())
        tokens = sum(v["tokens_sum"] for v in group.values())
        cost   = sum(v["cost_sum"]   for v in group.values())
        term   = sum(v["terminated"] for v in group.values())
        if n == 0:
            return {
                "n": 0, "exact_accuracy": None, "avg_f1": None,
                "avg_steps": None, "avg_tokens": None,
                "avg_cost_usd": None, "terminated_rate": None,
            }
        return {
            "n": n,
            "exact_accuracy":  round(exact / n, 4),
            "avg_f1":          round(f1s / n, 4),
            "avg_steps":       round(steps / n, 2),
            "avg_tokens":      round(tokens / n),
            "avg_cost_usd":    round(cost / n, 6),
            "terminated_rate": round(term / n, 4),
        }

    summary = {
        "split":       args.split,
        "n_questions": n,
        "seed":        args.seed,
        "timestamp":   timestamp,
        "overall": {
            "exact_accuracy": round(exact_acc, 4),
            "avg_f1":         round(avg_f1, 4),
            "finish_rate":    round(finish_rate, 4),
            "terminated_rate": round(term_rate, 4),
            "avg_step_count": round(avg_steps, 2),
            "avg_tokens":          round(avg_tokens),
            "avg_tokens_finished": round(avg_tokens_finished),
            "avg_cost_usd":        round(avg_cost, 6),
            "total_cost_usd": round(total_cost, 4),
            "total_cost_eur": round(total_cost * EUR_PER_USD, 4),
        },
        "by_type": {
            t: _agg({t: s})
            for t, s in type_stats.items()
        },
        "by_complexity": {
            "easy (2-3 nodes)":    _agg(easy_nodes),
            "complex (4-5 nodes)": _agg(complex_nodes),
            "by_num_nodes": {
                k: _agg({k: s})
                for k, s in sorted(complexity_stats.items())
            },
        },
    }

    print(f"{'═'*72}")
    print(f"RESULTS  (n={n}, split={args.split}, seed={args.seed})")
    print(f"  Exact accuracy : {exact_acc:.1%}")
    print(f"  Avg F1         : {avg_f1:.3f}")
    print(f"  Finish rate    : {finish_rate:.1%}")
    print(f"  Terminated     : {term_rate:.1%}")
    print(f"  Avg steps      : {avg_steps:.1f}")
    print(f"  Avg tokens     : {avg_tokens:.0f}  (finished only: {avg_tokens_finished:.0f})")
    print(f"  Total cost     : ${total_cost:.2f}  (~€{total_cost * EUR_PER_USD:.2f})")
    print(f"\nBY TYPE:")
    for t, s in summary["by_type"].items():
        print(f"  {t:15s}: n={s['n']:3d}  exact={s['exact_accuracy']:.1%}  f1={s['avg_f1']:.3f}")

    print(f"\nBY COMPLEXITY:")
    for label, s in [("easy   (2-3)", _agg(easy_nodes)), ("complex (4-5)", _agg(complex_nodes))]:
        if s["n"]:
            print(f"  {label}: n={s['n']:3d}  exact={s['exact_accuracy']:.1%}  f1={s['avg_f1']:.3f}")
    print(f"  per num_nodes:")
    for k, s in sorted(summary["by_complexity"]["by_num_nodes"].items()):
        print(f"    num_nodes={k}: n={s['n']:3d}  exact={s['exact_accuracy']:.1%}  f1={s['avg_f1']:.3f}")
    print(f"\nOutput: {csv_out}")

    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"        {json_out}")


if __name__ == "__main__":
    main()
