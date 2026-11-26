
import json
import logging
import operator

from dataclasses import asdict
from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

from django.conf import settings

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from knowledge.retrieval import RetrievedChunk

from .llm import (
    MAX_TOOL_ROUNDS,
    LLMResult,
    LLMUnavailable,
    execute_tool_calls,
    extract_text,
    get_chat_model,
)

from .prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    UNDERSTAND_SYSTEM,
    render_router_input,
    render_user_turn,
)

from .rag import (
    RagContext,
    retrieve_for_query,
    sources_for_api,
)

from .tools import build_tools


logger = logging.getLogger(__name__)


RECURSION_LIMIT = 20
MAX_STANDALONE_CHARS = 500

RETRIEVAL_INTENTS = {"policy", "both"}


# UNDERSTANDING MODEL
class Understanding(BaseModel):

    intent: Literal[
        "policy",
        "order",
        "both",
        "smalltalk",
    ] = Field(
        description="Which kind of help the newest message needs."
    )

    standalone_question: str = Field(
        description="The newest message rewritten so it is understandable without conversation history."
    )


def parse_understanding(response) -> Understanding | None:
    """
    Parse router response without using provider-specific
    structured output.
    """

    content = extract_text(response.content)

    if not content:
        return None

    # Handle markdown JSON fences.
    content = content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "")
        content = content.strip()

    # Handle accidental text surrounding JSON.
    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        return None

    content = content[start:end + 1]

    try:
        data = json.loads(content)
        return Understanding.model_validate(data)

    except (json.JSONDecodeError, ValidationError):
        logger.warning("Invalid router JSON response")
        return None


# AGENT STATE
class AgentState(TypedDict, total=False):

    question: str
    history: list

    intent: str
    standalone_question: str

    rag_status: str
    chunks: list

    messages: Annotated[list[AnyMessage], add_messages]

    tool_rounds: int
    trace: Annotated[list, operator.add]

    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]

    saw_usage: Annotated[bool, operator.or_]

    answer: str


# INITIAL STATE
def initial_state(history, message):

    return {
        "question": message,
        "history": [list(item) for item in history],

        "tool_rounds": 0,
        "trace": [],

        "input_tokens": 0,
        "output_tokens": 0,

        "saw_usage": False,
    }


# TOKEN USAGE
def _usage(message):

    if message is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "saw_usage": False,
        }

    usage = getattr(message, "usage_metadata", None) or {}

    return {
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "saw_usage": bool(usage),
    }


# RAG CONTEXT
def context_from_state(state):

    status = state.get("rag_status") or "not_needed"

    chunks = tuple(
        RetrievedChunk(**chunk)
        for chunk in state.get("chunks") or []
    )

    return RagContext(status, chunks)



# GRAPH BUILDER
def build_graph(user):

    """
    Build a graph for the authenticated user.

    User-specific tools are created here.
    User is never stored inside graph state.
    """

    base_model = get_chat_model()

    tools = build_tools(user)

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    # NODE 1: UNDERSTAND
    def understand(state):

        question = state["question"]

        fallback = {
            "intent": "both",
            "standalone_question": question[:MAX_STANDALONE_CHARS],
        }

        router_messages = [
            SystemMessage(
                content=(
                    UNDERSTAND_SYSTEM
                    + "\n\n"
                    + """
Return ONLY valid JSON.

Required format:

{
    "intent": "policy",
    "standalone_question": "..."
}

Allowed intent values:
- policy
- order
- both
- smalltalk

Do not use markdown.
Do not include explanations.
"""
                )
            ),

            HumanMessage(
                content=render_router_input(
                    state.get("history") or [],
                    question,
                )
            ),
        ]

        try:

            response = base_model.invoke(router_messages)

            parsed = parse_understanding(response)

        except Exception:

            logger.warning(
                "Router failed; falling back to intent=both",
                exc_info=True,
            )

            return fallback

        usage = _usage(response)

        if parsed is None:

            logger.warning(
                "Router returned invalid JSON; using fallback"
            )

            return {
                **fallback,
                **usage,
            }

        standalone = (
            parsed.standalone_question.strip()
            [:MAX_STANDALONE_CHARS]
        )

        return {
            "intent": parsed.intent,

            "standalone_question": (
                standalone
                or fallback["standalone_question"]
            ),

            **usage,
        }

    # NODE 2: RETRIEVE
    def retrieve(state):

        context = retrieve_for_query(
            state["standalone_question"]
        )

        return {
            "rag_status": context.status,

            "chunks": [
                asdict(chunk)
                for chunk in context.chunks
            ],
        }

    # NODE 3: PREPARE
    def prepare(state):

        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]

        for role, content in state.get("history") or []:

            if role == "user":

                messages.append(
                    HumanMessage(content=content)
                )

            else:

                messages.append(
                    AIMessage(content=content)
                )

        context = context_from_state(state)

        messages.append(
            HumanMessage(
                content=render_user_turn(
                    state["question"],
                    context,
                )
            )
        )

        return {
            "messages": messages
        }

    # NODE 4: AGENT
    def agent(state):

        model = base_model

        if state.get("intent") != "smalltalk":

            model = base_model.bind_tools(tools)

        try:

            response = model.invoke(
                state["messages"],
                timeout=settings.LLM_TIMEOUT,
            )

        except Exception as exc:

            logger.exception("LLM call failed")

            raise LLMUnavailable from exc

        return {
            "messages": [response],
            **_usage(response),
        }

    # NODE 5: TOOLS
    def tools_node(state):

        last_message = state["messages"][-1]

        calls = list(
            getattr(last_message, "tool_calls", None) or []
        )

        tool_messages, trace = execute_tool_calls(
            tools_by_name,
            calls,
        )

        return {
            "messages": tool_messages,

            "trace": trace,

            "tool_rounds": (
                state.get("tool_rounds", 0) + 1
            ),
        }

    # NODE 6: FINISH
    def finish(state):

        last_message = state["messages"][-1]

        text = extract_text(last_message.content)

        if not text:

            logger.error(
                "LLM returned an empty response"
            )

            raise LLMUnavailable(
                "empty response"
            )

        return {
            "answer": text
        }


    # NODE 7: GIVE UP
    def give_up(state):

        logger.error(
            "Tool loop exceeded maximum rounds: %s",
            MAX_TOOL_ROUNDS,
        )

        raise LLMUnavailable(
            "tool loop limit reached"
        )


    # ROUTING
    def after_understand(state):

        if state["intent"] in RETRIEVAL_INTENTS:

            return "retrieve"

        return "prepare"

    def after_agent(state):

        last_message = state["messages"][-1]

        calls = getattr(
            last_message,
            "tool_calls",
            None,
        )

        if calls:

            if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:

                return "give_up"

            return "tools"

        return "finish"

    # BUILD GRAPH
    graph = StateGraph(AgentState)

    graph.add_node("understand", understand)
    graph.add_node("retrieve", retrieve)
    graph.add_node("prepare", prepare)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools_node)
    graph.add_node("finish", finish)
    graph.add_node("give_up", give_up)

    graph.add_edge(START, "understand")

    graph.add_conditional_edges(
        "understand",
        after_understand,
        {
            "retrieve": "retrieve",
            "prepare": "prepare",
        },
    )

    graph.add_edge("retrieve", "prepare")

    graph.add_edge("prepare", "agent")

    graph.add_conditional_edges(
        "agent",
        after_agent,
        {
            "tools": "tools",
            "finish": "finish",
            "give_up": "give_up",
        },
    )

    graph.add_edge("tools", "agent")

    graph.add_edge("finish", END)

    graph.add_edge("give_up", END)

    return graph.compile()

# RUN AGENT
def run_agent(user, history, message) -> LLMResult:

    graph = build_graph(user)

    try:

        final = graph.invoke(
            initial_state(history, message),

            config={
                "recursion_limit": RECURSION_LIMIT,
            },
        )

    except GraphRecursionError as exc:

        logger.error(
            "Graph exceeded recursion limit"
        )

        raise LLMUnavailable(
            "graph recursion limit"
        ) from exc

    context = context_from_state(final)

    saw_usage = final.get("saw_usage")

    return LLMResult(

        text=final["answer"],

        model_name=settings.LLM_MODEL,

        prompt_version=PROMPT_VERSION,

        input_tokens=(
            final["input_tokens"]
            if saw_usage else None
        ),

        output_tokens=(
            final["output_tokens"]
            if saw_usage else None
        ),

        tool_trace=tuple(
            final.get("trace") or ()
        ),

        sources=tuple(
            sources_for_api(context)
        ),

        retrieval_status=context.status,

        intent=final.get("intent", ""),
    )
