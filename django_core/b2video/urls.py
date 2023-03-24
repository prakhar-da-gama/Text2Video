from django.urls import re_path as url
from . import views


urlpatterns = [
     url('api/enter_blog/', views.get_blog, name = 'write a video'),
]