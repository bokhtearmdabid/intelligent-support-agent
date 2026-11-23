from decimal import Decimal

from rest_framework import serializers

from .models import Order, OrderItem, Payment, Product


def money(value):
    return format(value, ".2f")


def mask_reference(value):
    """শেষ ৪ অক্ষর ছাড়া বাকি ঢেকে দেয়।"""
    if not value:
        return None
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


class ProductSerializer(serializers.ModelSerializer):
    in_stock = serializers.SerializerMethodField()

    class Meta:
        model = Product
        fields = ("sku", "name", "description", "price", "in_stock")
        read_only_fields = fields

    def get_in_stock(self, obj):
        return obj.stock > 0


class OrderItemSerializer(serializers.ModelSerializer):
    sku = serializers.CharField(source="product.sku", read_only=True)
    product_name = serializers.CharField(source="product.name", read_only=True)
    line_total = serializers.SerializerMethodField()

    class Meta:
        model = OrderItem
        fields = ("sku", "product_name", "quantity", "unit_price", "line_total")
        read_only_fields = fields

    def get_line_total(self, obj):
        return money(obj.line_total)


class PaymentSerializer(serializers.ModelSerializer):
    transaction_id = serializers.SerializerMethodField()

    class Meta:
        model = Payment
        fields = ("method", "status", "amount", "transaction_id", "paid_at")
        read_only_fields = fields

    def get_transaction_id(self, obj):
        return mask_reference(obj.transaction_id)


class OrderTotalMixin:
    def get_total_amount(self, obj):
        # obj.items.all() prefetch করা cache থেকে আসে: নতুন query নয়
        total = sum((i.line_total for i in obj.items.all()), Decimal("0.00"))
        return money(total)


class OrderListSerializer(OrderTotalMixin, serializers.ModelSerializer):
    item_count = serializers.SerializerMethodField()
    total_amount = serializers.SerializerMethodField()

    class Meta:
        model = Order
        fields = (
            "order_number", "status", "created_at",
            "estimated_delivery", "item_count", "total_amount",
        )
        read_only_fields = fields

    def get_item_count(self, obj):
        return len(obj.items.all())


class OrderDetailSerializer(OrderTotalMixin, serializers.ModelSerializer):
    items = OrderItemSerializer(many=True, read_only=True)
    payments = PaymentSerializer(many=True, read_only=True)
    total_amount = serializers.SerializerMethodField()

    class Meta:
        model = Order
        fields = (
            "order_number", "status", "created_at", "shipping_address",
            "carrier", "tracking_number", "estimated_delivery",
            "shipped_at", "delivered_at", "items", "payments", "total_amount",
        )
        read_only_fields = fields


class OrderStatusSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(
        source="get_status_display", read_only=True
    )
    payment_status = serializers.SerializerMethodField()

    class Meta:
        model = Order
        fields = (
            "order_number", "status", "status_display", "carrier",
            "tracking_number", "estimated_delivery", "shipped_at",
            "delivered_at", "payment_status",
        )
        read_only_fields = fields

    def get_payment_status(self, obj):
        payments = list(obj.payments.all())   # Meta ordering: নতুনটা আগে
        return payments[0].status if payments else None

class AgentPaymentSerializer(PaymentSerializer):
    class Meta(PaymentSerializer.Meta):
        fields = ("method", "status", "amount", "paid_at")   # transaction id একেবারেই নেই
        read_only_fields = fields


class AgentOrderDetailSerializer(OrderDetailSerializer):
    payments = AgentPaymentSerializer(many=True, read_only=True)

    class Meta(OrderDetailSerializer.Meta):
        fields = tuple(
            f for f in OrderDetailSerializer.Meta.fields if f != "shipping_address"
        )
        read_only_fields = fields