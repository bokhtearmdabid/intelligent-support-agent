import json
import logging
from dataclasses import dataclass
from functools import lru_cache

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, render_user_turn

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """Provider ব্যর্থ, ধীর বা অকেজো উত্তর দিয়েছে।"""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model_name: str = ""
    prompt_version: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_trace: tuple = ()


@lru_cache(maxsize=1)
def get_chat_model():
    if not settings.LLM_MODEL:
        raise ImproperlyConfigured("LLM_MODEL is not set in .env")

    kwargs = {
        "timeout": settings.LLM_TIMEOUT,
        "max_retries": settings.LLM_MAX_RETRIES,
        "max_tokens": settings.LLM_MAX_TOKENS,
    }

    if settings.LLM_TEMPERATURE is not None:
        kwargs["temperature"] = settings.LLM_TEMPERATURE

    return init_chat_model(
        model=settings.LLM_MODEL,
        model_provider=settings.LLM_PROVIDER,
        **kwargs,
    )


def _extract_text(content) -> str:
    """উত্তর কখনো স্ট্রিং, কখনো ব্লকের তালিকা: দুটোই সামলাই।"""

    if isinstance(content, str):
        return content.strip()

    parts = []

    for block in content or []:
        if isinstance(block, str):
            parts.append(block)

        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))

    return "".join(parts).strip()


MAX_TOOL_ROUNDS = 4          # সর্বোচ্চ কতবার tool-চক্র
MAX_CALLS_PER_ROUND = 3      # এক চক্রে সর্বোচ্চ কতটা tool call
TRACE_ARGS_CHARS = 200


def _error_json(code, message):
    return json.dumps({"error": code, "message": message}, ensure_ascii=False)


def _run_tool(tools_by_name, call):
    """একটা tool call চালায়। ফেরত: (ToolMessage-এর লেখা, সফল কি না)।"""
    tool = tools_by_name.get(call.get("name"))
    if tool is None:
        return _error_json("unknown_tool", "This tool does not exist."), False
    try:
        output = tool.invoke(call.get("args") or {})
    except ValidationError:
        return _error_json("invalid_arguments", "The tool arguments were invalid."), False
    except Exception:
        logger.exception("Tool %s failed", call.get("name"))
        return _error_json("tool_failed", "The tool could not complete the request."), False

    text = output if isinstance(output, str) else json.dumps(
        output, ensure_ascii=False, default=str
    )
    try:
        ok = "error" not in json.loads(text)
    except (ValueError, TypeError):
        ok = True
    return text, ok


def generate_reply(history, user_message, context=None, tools=None) -> LLMResult:
    """history: [(role, content), ...]; context: RagContext বা None; tools: build_tools(user)।"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    final_turn = (
        render_user_turn(user_message, context) if context is not None else user_message
    )
    messages.append(HumanMessage(content=final_turn))      # শুধুই HumanMessage

    model = get_chat_model()        # কনফিগ-ভুল এখানে জোরে ব্যর্থ হবে (try-র বাইরে)
    tools_by_name = {}
    if tools:
        tools_by_name = {t.name: t for t in tools}
        model = model.bind_tools(list(tools))

    trace = []
    in_total = out_total = 0
    saw_usage = False

    for _ in range(MAX_TOOL_ROUNDS + 1):
        try:
            response = model.invoke(messages, timeout=settings.LLM_TIMEOUT)
        except Exception as exc:
            logger.exception("LLM call failed")
            raise LLMUnavailable from exc

        usage = getattr(response, "usage_metadata", None) or {}
        if usage:
            saw_usage = True
            in_total += usage.get("input_tokens") or 0
            out_total += usage.get("output_tokens") or 0

        calls = list(getattr(response, "tool_calls", None) or [])
        if not calls:
            break                                   # চূড়ান্ত উত্তর

        messages.append(response)                   # LLM-এর মূল AIMessage হুবহু (Gemini-র signature সহ)
        for index, call in enumerate(calls):
            if index >= MAX_CALLS_PER_ROUND:
                content = _error_json("too_many_tool_calls", "Too many tool calls at once.")
                ok = False
            else:
                content, ok = _run_tool(tools_by_name, call)
            trace.append({
                "tool": str(call.get("name")),
                "args": json.dumps(call.get("args") or {}, ensure_ascii=False, default=str)[:TRACE_ARGS_CHARS],
                "ok": ok,
            })
            # প্রতিটা call-এর একটা করে ToolMessage লাগবেই, নইলে Gemini ত্রুটি দেয়
            messages.append(ToolMessage(
                content=content,
                tool_call_id=call.get("id") or "",
                name=call.get("name") or "",
            ))
    else:
        logger.error("Tool loop did not finish after %s rounds", MAX_TOOL_ROUNDS)
        raise LLMUnavailable("tool loop limit reached")

    text = _extract_text(response.content)
    if not text:
        logger.error("LLM returned an empty response")
        raise LLMUnavailable("empty response")

    return LLMResult(
        text=text,
        model_name=settings.LLM_MODEL,
        prompt_version=PROMPT_VERSION,
        input_tokens=in_total if saw_usage else None,
        output_tokens=out_total if saw_usage else None,
        tool_trace=tuple(trace),
    )
