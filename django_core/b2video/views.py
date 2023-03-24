import datetime
from django.shortcuts import get_object_or_404, redirect
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes,authentication_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.http import HttpResponse, HttpResponseRedirect
from rest_framework.parsers import JSONParser
from django.http import JsonResponse
from . import config
from .serializers import *
import json
# Create your views here.
@csrf_exempt
@api_view(['POST'])
def get_blog(request):
    response = {}
    if request.method == 'POST':
        jsonData = JSONParser().parse(request)
        config.title = jsonData['title']
        config.blog = jsonData['blog']
        from . import blog2video_core
        serializer = blog_title_serializer(data= jsonData)
        if serializer.is_valid():
            serializer.save()
            
            return  JsonResponse(serializer.data, safe=False)
    else:
        return HttpResponse("<h1>not_found</h1>")






