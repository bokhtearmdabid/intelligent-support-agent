import json
import re

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from shop.models import Order
from shop.policies import assess_return_eligibility
from shop.serializers import (
    AgentOrderDetailSerializer,
    OrderListSerializer,
    OrderStatusSerializer,
)
from shop.services import get_order_for_customer, orders_for_customer

ORDER_NUMBER_RE = re.compile(r"^ORD-[A-Z0-9]{8}$")
MAX_LIST = 10

_NOT_FOUND = ("order_not_found", "No order with this number was found on this customer's account.")
_BAD_NUMBER = ("invalid_order_number", "Order numbers look like ORD-XXXXXXXX (8 letters or digits).")


def _ok(data):
    return json.dumps(data, ensure_ascii=False, default=str)


def _error(code, message):
    return json.dumps({"error": code, "message": message}, ensure_ascii=False)


class OrderNumberArgs(BaseModel):
    order_number: str = Field(
        description="The order number the customer mentioned, for example ORD-AB12CD34."
    )


class ListOrdersArgs(BaseModel):
    limit: int = Field(default=5, description="How many recent orders to list (1 to 10).")


def build_tools(user):
    """এই user-এর জন্য বাঁধা tool-এর তালিকা। প্রতি request-এ নতুন করে।

    user কখনো LLM-এর আর্গুমেন্ট থেকে আসে না। এটাই পুরো নিরাপত্তার ভিত্তি।
    """

    def _load_order(raw_number):
        number = (raw_number or "").strip().upper()
        if not ORDER_NUMBER_RE.match(number):
            return None, _error(*_BAD_NUMBER)
        try:
            return get_order_for_customer(user, number), None
        except Order.DoesNotExist:
            # অন্যের order আর অস্তিত্বহীন order: হুবহু একই উত্তর
            return None, _error(*_NOT_FOUND)

    def list_my_orders(limit: int = 5) -> str:
        limit = max(1, min(int(limit), MAX_LIST))
        orders = list(orders_for_customer(user)[:limit])
        return _ok({
            "count": len(orders),
            "orders": OrderListSerializer(orders, many=True).data,
        })

    def get_order_status(order_number: str) -> str:
        order, error = _load_order(order_number)
        if error:
            return error
        return _ok(OrderStatusSerializer(order).data)

    def get_order_details(order_number: str) -> str:
        order, error = _load_order(order_number)
        if error:
            return error
        return _ok(AgentOrderDetailSerializer(order).data)

    def check_refund_eligibility(order_number: str) -> str:
        order, error = _load_order(order_number)
        if error:
            return error
        return _ok(assess_return_eligibility(order))

    return [
        StructuredTool.from_function(
            func=list_my_orders, name="list_my_orders", args_schema=ListOrdersArgs,
            description=(
                "List the logged-in customer's most recent orders (order number, status, "
                "date, item count, total). Use when the customer has not given an order "
                "number or asks about all their orders."
            ),
        ),
        StructuredTool.from_function(
            func=get_order_status, name="get_order_status", args_schema=OrderNumberArgs,
            description=(
                "Get the delivery status of one of the customer's orders: status, courier, "
                "tracking number, estimated delivery date, delivered date and payment status."
            ),
        ),
        StructuredTool.from_function(
            func=get_order_details, name="get_order_details", args_schema=OrderNumberArgs,
            description=(
                "Get the items, quantities, amounts and payments of one of the customer's "
                "orders. Use when the customer asks what they ordered or how much they paid."
            ),
        ),
        StructuredTool.from_function(
            func=check_refund_eligibility, name="check_refund_eligibility",
            args_schema=OrderNumberArgs,
            description=(
                "Check whether one of the customer's orders is still inside the return time "
                "window. This is NOT an approval: the support team decides every request."
            ),
        ),
    ]