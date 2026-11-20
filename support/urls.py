from django.urls import path
from rest_framework.routers import SimpleRouter

from .views import ChatView, ConversationViewSet

router = SimpleRouter()
router.register("conversations", ConversationViewSet, basename="conversation")

urlpatterns = [path("", ChatView.as_view(), name="chat")] + router.urls