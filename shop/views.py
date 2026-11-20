from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ReadOnlyModelViewSet

from .models import Product
from .serializers import (
    OrderDetailSerializer,
    OrderListSerializer,
    OrderStatusSerializer,
    ProductSerializer,
)
from .services import orders_for_customer


class ProductViewSet(ReadOnlyModelViewSet):
    """পাবলিক ক্যাটালগ: শুধু সক্রিয় পণ্য।"""
    queryset = Product.objects.filter(is_active=True).order_by("name")
    serializer_class = ProductSerializer
    permission_classes = [AllowAny]
    lookup_field = "sku"


class OrderViewSet(ReadOnlyModelViewSet):
    """লগইন করা customer-এর নিজের order।"""
    permission_classes = [IsAuthenticated]
    lookup_field = "order_number"

    def get_queryset(self):
        return orders_for_customer(self.request.user)

    def get_serializer_class(self):
        if self.action == "list":
            return OrderListSerializer
        if self.action == "order_status":
            return OrderStatusSerializer
        return OrderDetailSerializer

    @action(detail=True, methods=["get"], url_path="status")
    def order_status(self, request, order_number=None):
        order = self.get_object()
        return Response(self.get_serializer(order).data)