from django.db.models import Prefetch

from .models import Order, OrderItem


def orders_for_customer(user):

    if not user.is_authenticated:
        return Order.objects.none()

    return (
        Order.objects.filter(customer=user)
        .prefetch_related(
            Prefetch(
                "items",
                queryset=OrderItem.objects.select_related("product"),
            ),
            "payments",
        )
    )


def get_order_for_customer(user, order_number):
    return orders_for_customer(user).get(order_number=order_number)