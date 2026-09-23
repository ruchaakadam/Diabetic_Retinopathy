from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import render, redirect

from .forms import SignUpForm


def signup_view(request):

    if request.user.is_authenticated:
        return redirect("upload")

    if request.method == "POST":

        form = SignUpForm(request.POST)

        if form.is_valid():

            user = form.save()

            login(request, user)

            return redirect("upload")

    else:

        form = SignUpForm()

    return render(
        request,
        "registration/signup.html",
        {
            "form": form
        }
    )