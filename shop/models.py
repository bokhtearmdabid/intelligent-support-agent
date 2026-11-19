import secrets
import string

from django.conf import settings
from django.db import models
from django.db.models import F, Q, Sum
from django.utils import timezone


def generate_order_number():
    alphabet = string.ascii_uppercase + string.digits
    return "ORD-" + "".join(secrets.choice(alphabet) for _ in range(8))


class Product(models.Model):
    sku = models.CharField(max_length=40, unique=True)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=Q(price__gte=0), name="product_price_gte_0"
            ),
        ]

    def __str__(self):
        return f"{self.name} ({self.sku})"


class Order(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        CONFIRMED = "confirmed", "Confirmed"
        SHIPPED = "shipped", "Shipped"
        DELIVERED = "delivered", "Delivered"
        CANCELLED = "cancelled", "Cancelled"

    ALLOWED_TRANSITIONS = {
        Status.PENDING: {Status.CONFIRMED, Status.CANCELLED},
        Status.CONFIRMED: {Status.SHIPPED, Status.CANCELLED},
        Status.SHIPPED: {Status.DELIVERED},
        Status.DELIVERED: set(),
        Status.CANCELLED: set(),
    }

    order_number = models.CharField(
        max_length=20, unique=True, default=generate_order_number, editable=False
    )
    customer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="orders",
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    shipping_address = models.TextField()
    carrier = models.CharField(max_length=60, blank=True)
    tracking_number = models.CharField(max_length=60, blank=True)
    estimated_delivery = models.DateField(null=True, blank=True)
    shipped_at = models.DateTimeField(null=True, blank=True)
    delivered_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["customer", "-created_at"]),
        ]

    def __str__(self):
        return self.order_number

    @property
    def total_amount(self):
        result = self.items.aggregate(
            total=Sum(
                F("quantity") * F("unit_price"),
                output_field=models.DecimalField(max_digits=12, decimal_places=2),
            )
        )
        return result["total"] or 0

    def can_transition_to(self, new_status):
        return new_status in self.ALLOWED_TRANSITIONS.get(self.status, set())

    def transition_to(self, new_status):
        if not self.can_transition_to(new_status):
            raise ValueError(
                f"Order {self.order_number}: cannot change status "
                f"from '{self.status}' to '{new_status}'."
            )
        self.status = new_status
        now = timezone.now()
        if new_status == self.Status.SHIPPED:
            self.shipped_at = now
        elif new_status == self.Status.DELIVERED:
            self.delivered_at = now
        self.save()


class OrderItem(models.Model):
    order = models.ForeignKey(
        Order, on_delete=models.CASCADE, related_name="items"
    )
    product = models.ForeignKey(Product, on_delete=models.PROTECT)
    quantity = models.PositiveIntegerField()
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=Q(quantity__gt=0), name="orderitem_quantity_gt_0"
            ),
        ]

    @property
    def line_total(self):
        return self.quantity * self.unit_price

    def __str__(self):
        return f"{self.quantity} x {self.product.name}"


class Payment(models.Model):
    class Method(models.TextChoices):
        COD = "cod", "Cash on Delivery"
        BKASH = "bkash", "bKash"
        NAGAD = "nagad", "Nagad"
        CARD = "card", "Card"

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PAID = "paid", "Paid"
        FAILED = "failed", "Failed"
        REFUNDED = "refunded", "Refunded"

    order = models.ForeignKey(
        Order, on_delete=models.PROTECT, related_name="payments"
    )
    method = models.CharField(max_length=20, choices=Method.choices)
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    transaction_id = models.CharField(
        max_length=100, unique=True, null=True, blank=True
    )
    paid_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.order.order_number} - {self.method} - {self.status}"