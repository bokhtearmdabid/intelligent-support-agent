from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from .models import Customer


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = Customer
        fields = ("id", "email", "password", "first_name", "last_name", "phone")
        read_only_fields = ("id",)

    def validate_password(self, value):
        validate_password(value)   # Django-র password শক্ত কিনা যাচাই
        return value

    def create(self, validated_data):
        return Customer.objects.create_user(**validated_data)


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)


class CustomerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Customer
        fields = ("id", "email", "first_name", "last_name", "phone", "role")
        read_only_fields = fields