from rest_framework import serializers
from .models import *;
class blog_title_serializer(serializers.Serializer):
    title = serializers.CharField(max_length = 30)
    blog = serializers.CharField(max_length = 50)
    def create(self, validated_data):

        return blogModel.objects.create(**validated_data)