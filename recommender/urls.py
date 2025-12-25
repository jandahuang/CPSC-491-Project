from django.urls import path
from . import views

app_name = 'recommender'

urlpatterns = [
    path('', views.main, name='main'),
    path('api/search/', views.search_movies, name='search_movies'),
    path('api/model-status/', views.model_status, name='model_status'),
    path('api/health/', views.health_check, name='health_check'),
]
