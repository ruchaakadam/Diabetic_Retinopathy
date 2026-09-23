from django.urls import path
from django.contrib.auth import views as auth_views

from .views import upload_image
from .auth_views import signup_view


urlpatterns = [

    # ---------------------------------------------------------
    # AUTHENTICATION
    # ---------------------------------------------------------

    path(
        "login/",
        auth_views.LoginView.as_view(
            template_name="registration/login.html"
        ),
        name="login"
    ),

    path(
        "signup/",
        signup_view,
        name="signup"
    ),

    path(
        "logout/",
        auth_views.LogoutView.as_view(),
        name="logout"
    ),

    # ---------------------------------------------------------
    # RETINAAI SCREENING
    # ---------------------------------------------------------

    path(
        "",
        upload_image,
        name="upload"
    ),
]