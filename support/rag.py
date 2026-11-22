import logging
from dataclasses import dataclass

from knowledge.embeddings import EmbeddingError
from knowledge.retrieval import search_knowledge

logger = logging.getLogger(__name__)

SHORT_MESSAGE_CHARS = 40
MAX_QUERY_CHARS = 1000


@dataclass(frozen=True)
class RagContext:
    status: str            # "ok" | "empty" | "unavailable"
    chunks: tuple = ()


def build_retrieval_query(history, message):
    """ছোট follow-up ("আর কার্ডে?") হলে আগের প্রশ্নের সাথে জুড়ে খুঁজি।"""
    if history and len(message) < SHORT_MESSAGE_CHARS:
        last_user = next((c for role, c in reversed(history) if role == "user"), None)
        if last_user:
            return f"{last_user}\n{message}"[:MAX_QUERY_CHARS]
    return message[:MAX_QUERY_CHARS]


def retrieve_context(history, message):
    query = build_retrieval_query(history, message)
    try:
        chunks = search_knowledge(query)
    except EmbeddingError:
        logger.warning("Knowledge search unavailable; answering without excerpts")
        return RagContext("unavailable")
    return RagContext("ok" if chunks else "empty", tuple(chunks))


def sources_for_api(context):
    """Customer-কে দেখানোর উৎস: শুধু ডকুমেন্ট ও section, কোনো distance বা বিষয়বস্তু নয়।"""
    seen, sources = set(), []
    for chunk in context.chunks:
        heading = chunk.heading_path
        section = heading.split(" > ", 1)[1] if " > " in heading else heading
        key = (chunk.document_title, section)
        if key not in seen:
            seen.add(key)
            sources.append({"document": chunk.document_title, "section": section})
    return sources