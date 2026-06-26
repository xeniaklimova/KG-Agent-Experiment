from __future__ import annotations

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_azure_ai.chat_models import AzureChatOpenAI
from langchain_neo4j import Neo4jGraph

from kgagent import tools, prompts
from kgagent.agent import build_app, ask, AgentState
from kgagent.config import MAX_STEPS, TIME_HARD_CAP, TOKEN_HARD_CAP
from kgagent.prompts import SYSTEM_PROMPT

graph_db = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    database="enrich3",
)
graph_db.refresh_schema()
tools.set_graph_db(graph_db)

llm = ChatMistralAI(
    model=os.environ["MISTRAL_MODEL_NAME"],
    base_url=os.environ["MISTRAL_BASE_URL"],
    api_key=os.environ["MISTRAL_API_KEY"],
)

# llm = AzureChatOpenAI(
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     api_key=os.getenv("OPENAI_API_KEY"),
#     azure_deployment="gpt-4.1",
#     api_version="2025-03-01-preview",
#     temperature=0,
# )

schema_annotations = prompts.SCHEMA_ANNOTATIONS["en3"]
app = build_app(llm=llm, schema_annotations=schema_annotations, owner_module=sys.modules[__name__])
