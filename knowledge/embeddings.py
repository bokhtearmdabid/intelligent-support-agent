import logging
import math
from functools import lru_cache

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Embedding তৈরি ব্যর্থ বা অগ্রহণযোগ্য ফলাফল।"""


@lru_cache(maxsize=1)
def _client():
    if not settings.EMBEDDING_MODEL:
        raise ImproperlyConfigured("EMBEDDING_MODEL is not set in .env")
    return GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)


def _clean(vector):
    """মাত্রা যাচাই + দৈর্ঘ্য ১-এ আনা (normalize)।"""
    if len(vector) != settings.EMBEDDING_DIMENSIONS:
        raise EmbeddingError(
            f"Expected {settings.EMBEDDING_DIMENSIONS} dimensions, got {len(vector)}"
        )
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise EmbeddingError("Received an all-zero embedding")
    return [x / norm for x in vector]


def embed_documents(texts):
    if not texts:
        return []
    client = _client()          # কনফিগ-ভুল এখানে জোরে ব্যর্থ হবে, try-এর বাইরে
    try:
        vectors = client.embed_documents(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=settings.EMBEDDING_DIMENSIONS,
        )
    except Exception as exc:
        logger.exception("Embedding request failed")
        raise EmbeddingError("Embedding request failed") from exc

    if len(vectors) != len(texts):
        raise EmbeddingError("Embedding count does not match input count")
    return [_clean(v) for v in vectors]


def embed_query(text):
    client = _client()
    try:
        vector = client.embed_query(
            text,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=settings.EMBEDDING_DIMENSIONS,
        )
    except Exception as exc:
        logger.exception("Query embedding failed")
        raise EmbeddingError("Query embedding failed") from exc
    return _clean(vector)