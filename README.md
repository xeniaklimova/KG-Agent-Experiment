# KG Agent

A LangGraph-based agentic system for multi-hop reasoning over the POLE knowledge graph stored in Neo4j. The agent answers natural language questions by iteratively traversing the graph using a structured tool set.

## Abstract

## Research Questions


## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd KG-Agent-Experiment
   ```

2. Install dependencies:

   This project uses uv as a package manager.
   
   ```bash
   uv sync
   ```

4. Create a `.env` file with the following variables:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password

   # Mistral (direct API)
   MISTRAL_MODEL_NAME=mistral-medium-latest
   MISTRAL_BASE_URL=https://api.mistral.ai/v1
   MISTRAL_API_KEY=your_mistral_api_key

   # GPT-4.1 via Microsoft Foundry
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   OPENAI_API_KEY=your_azure_api_key
   ```

## Running Evaluations

```bash
uv run eval/eval_runner.py \
  --agent mistral_base \
  --split mixed \
  --n 70 \
  --seed 88 \
  --workers 4
```

**Arguments:**
- `--agent` — script name without `.py` (e.g. `mistral_base`, `mistral_en1`, ..., `mistral_en4`)
- `--split` — `mixed` (equal train/test sample), `train`, or `test`
- `--n` — number of questions to evaluate
- `--seed` — random seed for reproducibility
- `--workers` — number of parallel calls

Results are written to `results/` as both `.csv` (per-question) and `.json` (summary statistics).

## Graph Configurations

| Config | Database | Enrichment |
|---|---|---|
| `mistral_base` | originalpole1 | None |
| `mistral_en1` | enrichrelprop | Relationship property descriptions |
| `mistral_en2` | enrichedge | Shortcut edges |
| `mistral_en3` | enrich3 | Neighbor information (per-node connectivity summaries) |
| `mistral_en4` | allenrich | Relationship descriptions + neighbor information |

## Agent Tools

| Tool | Description |
|---|---|
| `find_nodes` | Locate anchor nodes by name or property value |
| `explore` | Discover connections or traverse a relationship from a node set |
| `node_feature` | Retrieve a single property for a set of nodes |
| `filter_by_constraint` | Filter or rank a candidate set by property value |
| `count_nodes` | Count nodes reachable via a relationship chain |
| `set_ops` | Intersect or union two node sets |
| `finish` | Submit the final structured answer |


## Project Structure

```
KG-Agent-Experiment/
├── kgagent/            # Core agent package
│   ├── config.py           # All constants (step limits, token budgets, etc.)
│   ├── tools.py            # graph tools (find_nodes, explore, filter_by_constraint, etc)
│   ├── prompts.py          # SYSTEM_PROMPT + SCHEMA_ANNOTATIONS per graph config
│   └── agent.py            # AgentState, graph construction, build_app(), ask()
│
├── scripts/                # Thin launchers — one per graph configuration
│   ├── mistral_base.py     # Base graph (no enrichment)
│   ├── mistral_en1.py      # + Relationship property descriptions
│   ├── mistral_en2.py      # + Shortcut edges
│   ├── mistral_en3.py      # + Neighbor information (node-level connectivity summaries)
│   └── mistral_en4.py      # + Relationship descriptions & neighbor information (combined)
│
├── eval/
│   ├── eval_runner.py      # Evaluation harness (old version)
│   └── eval_thread.py      # Per-question threaded execution (use this one)
│
├── data_prep/              # Graph enrichment scripts
│   ├── enrich_relprop.py   # Adds relationship description properties (EN1)
│   ├── enrich_edge.py      # Adds shortcut edges (EN2)
│   ├── enrich_nodedeg.py   # Adds neighbor_info node property (EN3)
│   └── enrich_allenrich.py # Combines EN1 + EN3 enrichments (EN4)
│
├── zogra/data/             # Evaluation question set ZOGRASCOPE, already pre-processed and ready for eval 
│   ├── zograscope_length_train_v1_answered_v3.csv
│   └── zograscope_length_test_v1_answered_v3.csv
│
├── pyproject.toml
└── uv.lock
```
