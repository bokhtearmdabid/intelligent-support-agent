from unittest.mock import patch

from django.conf import settings
from django.test import TestCase, override_settings

from knowledge.models import KnowledgeChunk, KnowledgeDocument
from knowledge.retrieval import search_knowledge

DIM = settings.EMBEDDING_DIMENSIONS
EMBED_QUERY = "knowledge.retrieval.embed_query"


def vec(*head):
    return list(head) + [0.0] * (DIM - len(head))


def make_chunk(doc, index, content, vector):
    return KnowledgeChunk.objects.create(
        document=doc, chunk_index=index, heading_path=f"Doc > {content}",
        content=content, embedding=vector, embedding_model="t",
    )


class RetrievalTests(TestCase):
    def setUp(self):
        self.doc = KnowledgeDocument.objects.create(
            source_name="a.md", title="A", status="ready", is_active=True
        )
        make_chunk(self.doc, 0, "near", vec(1, 0))      # query-র সাথে distance 0.0
        make_chunk(self.doc, 1, "mid", vec(0.8, 0.6))   # distance 0.2
        make_chunk(self.doc, 2, "far", vec(0, 1))       # distance 1.0

    def search(self, **kwargs):
        with patch(EMBED_QUERY, return_value=vec(1, 0)):
            return search_knowledge("question", **kwargs)

    def test_orders_by_distance_and_applies_threshold(self):
        results = self.search(k=5, max_distance=0.5)
        self.assertEqual([r.content for r in results], ["near", "mid"])
        self.assertAlmostEqual(results[0].distance, 0.0, places=5)
        self.assertAlmostEqual(results[1].distance, 0.2, places=5)

    def test_k_limits_results(self):
        self.assertEqual(len(self.search(k=1, max_distance=2)), 1)

    @override_settings(RAG_MAX_DISTANCE=0.1)
    def test_default_threshold_comes_from_settings(self):
        self.assertEqual([r.content for r in self.search()], ["near"])

    def test_inactive_and_failed_documents_are_excluded(self):
        self.doc.is_active = False
        self.doc.save()
        self.assertEqual(self.search(k=5, max_distance=2), [])
        self.doc.is_active = True
        self.doc.status = "failed"
        self.doc.save()
        self.assertEqual(self.search(k=5, max_distance=2), [])

    def test_blank_query_makes_no_api_call(self):
        with patch(EMBED_QUERY) as embed:
            self.assertEqual(search_knowledge("   "), [])
        embed.assert_not_called()