import logging
from dataclasses import dataclass
from functools import lru_cache

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .prompts import PROMPT_VERSION, SYSTEM_PROMPT

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


def generate_reply(history, user_message) -> LLMResult:
    """history: [(role, content), ...] পুরনো থেকে নতুন ক্রমে।"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))   # শুধুই HumanMessage

    model = get_chat_model()   # কনফিগ-ভুল এখানে জোরে ব্যর্থ হবে (৫০০)

    try:
        response = model.invoke(
        messages,
        timeout=60,
    )

    except Exception as exc:
        logger.exception("LLM call failed")
        raise LLMUnavailable from exc

    text = _extract_text(response.content)
    if not text:
        logger.error("LLM returned an empty response")
        raise LLMUnavailable("empty response")

    usage = getattr(response, "usage_metadata", None) or {}
    return LLMResult(
        text=text,
        model_name=settings.LLM_MODEL,
        prompt_version=PROMPT_VERSION,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )