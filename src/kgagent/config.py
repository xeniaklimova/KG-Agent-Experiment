from __future__ import annotations

# Properties searched per label for find_nodes (text= mode).
# Order matters — first property is tried first.
NAME_PROPERTIES: dict[str, list[str]] = {
    "Person":    ["name", "surname"],
    "Officer":   ["name", "surname", "rank"],
    "Crime":     ["type", "last_outcome"],
    "Vehicle":   ["make", "model", "reg"],
    "Location":  ["address", "postcode"],
    "Object":    ["type", "description"],
    "PhoneCall": ["call_type", "call_date"],
    "Phone":     ["phoneNo"],
    "Email":     ["email_address"],
    "Area":      ["areaCode"],
    "PostCode":  ["code"],
}

# How many entities before we switch to count-only mode.
ENTITY_LIST_LIMIT = 30
TOOL_MSG_MAX_CHARS = 5000  # cap per explore traversal response to prevent context explosion
CONTEXT_KEEP_HEAD = 4   # SystemMessage(prompt) + HumanMessage(q) + SystemMessage(schema) + first AI turn
CONTEXT_KEEP_TAIL = 12  # last ~7 complete tool-call exchanges

DISCOVERY_RELATION_LIMIT = 8  # max relation-type rows shown per node group in discovery mode

# Hard cutoffs — when reached the agent receives a strong text nudge ([FORCED FINISH] / [EXPLORE DISABLED]) to call finish().
HARD_FINISH_STEP  = 20
TOKEN_BUDGET      = 60000
MAX_EXPLORE_CALLS = 14
MAX_STEPS         = 24
TOKEN_HARD_CAP    = 90000
TIME_HARD_CAP     = 100
PRESSURE_STEP     = 17
CHECKPOINT_STEP   = 10
STREAK_TOOL              = "explore"
STREAK_THRESHOLD         = 3
STREAK_EXCLUDE_THRESHOLD = 8
