from django.urls import path
from ml import views

urlpatterns = [
    path('', views.Data.as_view(),name='data')
]