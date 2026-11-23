import json
from unittest.mock import patch

from django.core.cache import cache
from rest_framework.test import APITestCase

from accounts.models import Customer
from shop.models import Order
from support.llm import LLMResult
from support.models import ChatMessage
from support.rag import RagContext


class ChatToolsBindingTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.rahim = Customer.objects.create_user("rahim@test.com", "Pass!12345")
        cls.karim = Customer.objects.create_user("karim@test.com", "Pass!12345")
        cls.rahim_order = Order.objects.create(customer=cls.rahim, shipping_address="x")
        cls.karim_order = Order.objects.create(customer=cls.karim, shipping_address="y")

    def setUp(self):
        cache.clear()
        self.client.force_authenticate(self.rahim)

    def _post(self, result):
        with patch("support.views.retrieve_context", return_value=RagContext("empty")), \
             patch("support.views.generate_reply", return_value=result) as gen:
            response = self.client.post("/api/chat/", {"message": "hi"}, format="json")
        return response, gen

    def test_tools_given_to_the_llm_are_bound_to_the_logged_in_customer(self):
        _, gen = self._post(LLMResult(text="ok", prompt_version="v3"))
        tools = gen.call_args.kwargs["tools"]
        listing = json.loads(next(t for t in tools if t.name == "list_my_orders").invoke({}))
        numbers = {o["order_number"] for o in listing["orders"]}
        self.assertEqual(numbers, {self.rahim_order.order_number})
        self.assertNotIn(self.karim_order.order_number, numbers)

    def test_tool_trace_is_stored_but_never_returned_to_the_customer(self):
        trace = ({"tool": "get_order_status", "args": '{"order_number": "ORD-X"}', "ok": True},)
        response, _ = self._post(LLMResult(text="ok", prompt_version="v3", tool_trace=trace))
        self.assertEqual(response.status_code, 201)
        self.assertNotIn("get_order_status", response.content.decode())
        saved = ChatMessage.objects.get(role="assistant")
        self.assertEqual(saved.tool_calls[0]["tool"], "get_order_status")