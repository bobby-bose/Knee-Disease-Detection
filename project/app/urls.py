from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home, name='index'),
    path('register/', views.RegisterUserView.as_view(), name='register'),
    path('', views.LoginUserView.as_view(), name='login'),
    path('logout/', views.LogoutUserView.as_view(), name='logout'),

    path('preprocess_images/', views.preprocess_images, name='preprocess_images'),
    path('model_training/', views.model_training, name='model_training'),
    path('knee_identification/', views.knee_identification, name='knee_identification'),

]

