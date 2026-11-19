from django.contrib.auth import authenticate, login, logout
from rest_framework import status
from rest_framework.generics import CreateAPIView, RetrieveAPIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from .serializers import CustomerSerializer, LoginSerializer, RegisterSerializer


class RegisterView(CreateAPIView):
    serializer_class = RegisterSerializer
    permission_classes = [AllowAny]


class LoginView(APIView):
    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "login"

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = authenticate(
            request,
            username=serializer.validated_data["email"],
            password=serializer.validated_data["password"],
        )
        if user is None:
            return Response(
                {"detail": "Invalid email or password."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        login(request, user)   # Session তৈরি, cookie সেট
        return Response(CustomerSerializer(user).data)


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        logout(request)
        return Response(status=status.HTTP_204_NO_CONTENT)


class MeView(RetrieveAPIView):
    serializer_class = CustomerSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user