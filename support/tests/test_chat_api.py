from unittest.mock import patch

from django.core.cache import cache
from rest_framework.test import APIClient, APITestCase

from accounts.models import Customer
from support.llm import LLMResult, LLMUnavailable
from support.models import ChatMessage, SupportConversation
from support.services import recent_history

GENERATE = "support.views.generate_reply"


def fake_result(text="Hello from Aria"):
    return LLMResult(text=text, model_name="test-model", prompt_version="v1",
                     input_tokens=11, output_tokens=7)


class ChatApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.rahim = Customer.objects.create_user("rahim@test.com", "Pass!12345")
        cls.karim = Customer.objects.create_user("karim@test.com", "Pass!12345")

    def setUp(self):
        cache.clear()

    # ---------- Authentication ----------
    def test_anonymous_cannot_chat(self):
        with patch(GENERATE) as gen:
            response = self.client.post("/api/chat/", {"message": "hi"}, format="json")
        self.assertEqual(response.status_code, 403)
        gen.assert_not_called()

    def test_csrf_is_enforced_for_session_auth(self):
        client = APIClient(enforce_csrf_checks=True)
        client.force_login(self.rahim)
        with patch(GENERATE) as gen:
            response = client.post("/api/chat/", {"message": "hi"}, format="json")
        self.assertEqual(response.status_code, 403)
        gen.assert_not_called()

    # ---------- Happy path ----------
    def test_new_conversation_is_created_and_saved(self):
        self.client.force_authenticate(self.rahim)
        with patch(GENERATE, return_value=fake_result()):
            response = self.client.post(
                "/api/chat/", {"message": "Where is my order?"}, format="json"
            )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["reply"], "Hello from Aria")

        conv = SupportConversation.objects.get(public_id=response.data["conversation_id"])
        self.assertEqual(conv.customer, self.rahim)
        roles = list(conv.messages.values_list("role", flat=True))
        self.assertEqual(roles, ["user", "assistant"])
        assistant = conv.messages.last()
        self.assertEqual((assistant.input_tokens, assistant.output_tokens), (11, 7))

    def test_follow_up_sends_previous_messages_as_history(self):
        self.client.force_authenticate(self.rahim)
        with patch(GENERATE, return_value=fake_result("first reply")):
            first = self.client.post("/api/chat/", {"message": "first"}, format="json")
        cid = first.data["conversation_id"]

        with patch(GENERATE, return_value=fake_result("second reply")) as gen:
            self.client.post(
                "/api/chat/", {"message": "second", "conversation_id": cid},
                format="json",
            )
        history, message = gen.call_args.args
        self.assertEqual(history, [("user", "first"), ("assistant", "first reply")])
        self.assertEqual(message, "second")

    # ---------- Ownership ----------
    def test_other_customers_conversation_returns_404_and_skips_llm(self):
        conv = SupportConversation.objects.create(customer=self.karim, title="x")
        self.client.force_authenticate(self.rahim)
        with patch(GENERATE) as gen:
            response = self.client.post(
                "/api/chat/",
                {"message": "hi", "conversation_id": str(conv.public_id)},
                format="json",
            )
        self.assertEqual(response.status_code, 404)
        gen.assert_not_called()          # কোনো খরচ নয়, কোনো ফাঁস নয়
        self.assertEqual(conv.messages.count(), 0)

    # ---------- Validation ----------
    def test_invalid_input_is_rejected_before_llm(self):
        self.client.force_authenticate(self.rahim)
        bad_payloads = [
            {"message": ""},
            {"message": "    "},
            {"message": "x" * 2001},
            {"message": "hi", "conversation_id": "not-a-uuid"},
            {},
        ]
        with patch(GENERATE) as gen:
            for payload in bad_payloads:
                response = self.client.post("/api/chat/", payload, format="json")
                self.assertEqual(response.status_code, 400, payload)
        gen.assert_not_called()

    # ---------- Failure handling ----------
    def test_llm_failure_returns_503_and_saves_nothing(self):
        self.client.force_authenticate(self.rahim)
        with patch(GENERATE, side_effect=LLMUnavailable()):
            response = self.client.post("/api/chat/", {"message": "hi"}, format="json")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("Traceback", response.content.decode())
        self.assertEqual(SupportConversation.objects.count(), 0)
        self.assertEqual(ChatMessage.objects.count(), 0)


class HistoryTests(APITestCase):
    def setUp(self):
        self.user = Customer.objects.create_user("h@test.com", "Pass!12345")
        self.conv = SupportConversation.objects.create(customer=self.user)

    def _fill(self, pairs):
        for i in range(pairs):
            ChatMessage.objects.create(conversation=self.conv, role="user", content=f"u{i}")
            ChatMessage.objects.create(conversation=self.conv, role="assistant", content=f"a{i}")

    def test_history_is_limited_and_starts_with_user(self):
        self._fill(15)                                   # ৩০টা মেসেজ
        history = recent_history(self.conv, 20)
        self.assertEqual(len(history), 20)
        self.assertEqual(history[0], ("user", "u5"))
        self.assertEqual(history[-1], ("assistant", "a14"))

    def test_leading_assistant_message_is_dropped(self):
        self._fill(2)                                    # u0 a0 u1 a1
        history = recent_history(self.conv, 3)           # a0 u1 a1 → a0 বাদ
        self.assertEqual(history, [("user", "u1"), ("assistant", "a1")])


class ConversationReadApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.rahim = Customer.objects.create_user("rahim@test.com", "Pass!12345")
        cls.karim = Customer.objects.create_user("karim@test.com", "Pass!12345")
        cls.rahim_conv = SupportConversation.objects.create(customer=cls.rahim, title="mine")
        cls.karim_conv = SupportConversation.objects.create(customer=cls.karim, title="theirs")
        ChatMessage.objects.create(
            conversation=cls.rahim_conv, role="assistant", content="hello",
            model_name="secret-model", input_tokens=99,
        )

    def setUp(self):
        cache.clear()
        self.client.force_authenticate(self.rahim)

    def test_list_shows_only_own_conversations(self):
        response = self.client.get("/api/chat/conversations/")
        ids = [c["conversation_id"] for c in response.data["results"]]
        self.assertEqual(ids, [str(self.rahim_conv.public_id)])

    def test_other_customers_conversation_returns_404(self):
        response = self.client.get(f"/api/chat/conversations/{self.karim_conv.public_id}/")
        self.assertEqual(response.status_code, 404)

    def test_internal_fields_are_not_exposed(self):
        response = self.client.get(f"/api/chat/conversations/{self.rahim_conv.public_id}/")
        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertNotIn("secret-model", body)
        self.assertNotIn("input_tokens", body)
        self.assertEqual(response.data["messages"][0]["content"], "hello")

    def test_conversations_are_read_only(self):
        url = f"/api/chat/conversations/{self.rahim_conv.public_id}/"
        self.assertEqual(self.client.delete(url).status_code, 405)