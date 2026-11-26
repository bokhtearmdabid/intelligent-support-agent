import json
from unittest.mock import patch

from django.core.cache import cache
from rest_framework.test import APITestCase

from accounts.models import Customer
from support.llm import LLMResult
from support.models import ChatMessage


class ChatAgentWiringTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.rahim = Customer.objects.create_user("rahim@test.com", "Pass!12345")
        cls.karim = Customer.objects.create_user("karim@test.com", "Pass!12345")

    def setUp(self):
        cache.clear()
        self.client.force_authenticate(self.rahim)

    def _post(self, payload, result=None):
        result = result or LLMResult(text="ok", prompt_version="v4")
        with patch("support.views.run_agent", return_value=result) as run:
            response = self.client.post("/api/chat/", payload, format="json")
        return response, run

    def test_agent_runs_for_the_logged_in_customer(self):
        _, run = self._post({"message": "hi"})
        self.assertEqual(run.call_args.args[0], self.rahim)

    def test_request_body_cannot_choose_the_customer(self):
        _, run = self._post({"message": "hi", "customer_id": self.karim.id,
                             "user": self.karim.id, "email": "karim@test.com"})
        self.assertEqual(run.call_args.args[0], self.rahim)

    def test_tool_trace_is_stored_but_never_returned_to_the_customer(self):
        trace = ({"tool": "get_order_status", "args": '{"order_number": "ORD-X"}', "ok": True},)
        response, _ = self._post(
            {"message": "hi"}, LLMResult(text="ok", prompt_version="v4", tool_trace=trace))
        self.assertEqual(response.status_code, 201)
        self.assertNotIn("get_order_status", response.content.decode())
        self.assertEqual(ChatMessage.objects.get(role="assistant").tool_calls[0]["tool"],
                         "get_order_status")