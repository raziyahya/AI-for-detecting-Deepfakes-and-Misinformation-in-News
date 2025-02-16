from django.db import models

# Create your models here.
class Login(models.Model):
    Username=models.CharField(max_length=50)
    Password=models.CharField(max_length=50)
    Type=models.CharField(max_length=50)

class User(models.Model):
    LOGIN=models.ForeignKey(Login,on_delete=models.CASCADE)
    Name=models.CharField(max_length=50)
    DOB=models.DateField()
    Gender=models.CharField(max_length=12)
    Email=models.CharField(max_length=50)
    Phoneno=models.BigIntegerField()
    Image=models.FileField()

