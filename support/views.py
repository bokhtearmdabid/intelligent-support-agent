
from django.conf import settings

from rest_framework.exceptions import NotFound
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework.viewsets import ReadOnlyModelViewSet

from .graph import run_agent
from .llm import LLMUnavailable
from .models import SupportConversation
from .serializers import (
    ChatRequestSerializer,
    ConversationDetailSerializer,
    ConversationListSerializer,
)
from .services import (
    conversations_for_customer,
    get_conversation_for_customer,
    recent_history,
    save_exchange,
)


class ChatView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "chat"

    def post(self, request):
        # Validate incoming request
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        message = serializer.validated_data["message"]
        conversation_id = serializer.validated_data.get("conversation_id")

        conversation = None
        history = []

        # Load existing conversation and verify ownership
        if conversation_id:
            try:
                conversation = get_conversation_for_customer(
                    request.user,
                    conversation_id,
                )
            except SupportConversation.DoesNotExist:
                # Same response for nonexistent and unauthorized conversations
                raise NotFound()

            history = recent_history(
                conversation,
                settings.CHAT_HISTORY_LIMIT,
            )

        # Ownership verification completed.
        # Now safely execute the agent graph.
        try:
            result = run_agent(
                request.user,
                history,
                message,
            )

        except LLMUnavailable:
            return Response(
                {
                    "detail": (
                        "The assistant is temporarily unavailable. "
                        "Please try again in a moment."
                    )
                },
                status=503,
            )

        # Save user message, assistant response and agent metadata
        conversation = save_exchange(
            request.user,
            conversation,
            message,
            result,
            sources=result.sources,
            retrieval_status=result.retrieval_status,
            tool_trace=result.tool_trace,
            intent=result.intent,
        )

        # Return API response
        return Response(
            {
                "conversation_id": conversation.public_id,
                "reply": result.text,
                "sources": list(result.sources),
            },
            status=201,
        )


class ConversationViewSet(ReadOnlyModelViewSet):
    permission_classes = [IsAuthenticated]
    lookup_field = "public_id"

    def get_queryset(self):
        qs = conversations_for_customer(self.request.user)

        if self.action == "retrieve":
            qs = qs.prefetch_related("messages")

        return qs

    def get_serializer_class(self):
        if self.action == "retrieve":
            return ConversationDetailSerializer

        return ConversationListSerializer
