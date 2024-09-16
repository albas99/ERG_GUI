from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name = "index"),
    path("simple_upload", views.simple_upload, name = "simple_upload"),
    path("model_upload", views.model_upload, name = "model_upload"),
    path("get_selection", views.get_selection, name = "get_selection"),
]