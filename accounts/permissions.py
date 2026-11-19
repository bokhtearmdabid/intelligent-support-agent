from rest_framework.permissions import BasePermission

from .models import Customer


class IsCustomer(BasePermission):
    def has_permission(self, request, view):
        return (
            request.user.is_authenticated
            and request.user.role == Customer.Role.CUSTOMER
        )


class IsSupportStaff(BasePermission):
    def has_permission(self, request, view):
        return (
            request.user.is_authenticated
            and request.user.role in (Customer.Role.SUPPORT_AGENT, Customer.Role.ADMIN)
        )