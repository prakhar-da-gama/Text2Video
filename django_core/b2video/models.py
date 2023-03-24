from django.db import models

# Create your models here.
class blogModel(models.Model):
    title = models.CharField(max_length=255)
    
    blog = models.TextField()
    