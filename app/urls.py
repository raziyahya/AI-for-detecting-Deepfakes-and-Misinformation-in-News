"""Deepfake URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from Deepfake import settings

from . import views
urlpatterns = [
    path('', views.main),
    path('login', views.login),
    path('signup', views.signup),
    path('reg_code', views.reg_code),
    path('userhome', views.userhome),
    path('predict_image', views.predict_image),
    path('predict_imagefn', views.predict_imagefn),
    path('predict_video', views.predict_video),
    path('predict_audio', views.predict_audio),
    path('predict_text', views.predict_text),
    path('predict_text_post', views.predict_text_post),
    path('predict_audiofn', views.predict_audiofn),
    path('predict_videofn', views.predict_videofn),
]
