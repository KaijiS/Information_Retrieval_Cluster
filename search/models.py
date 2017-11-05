from django.db import models
from django import forms

class Paper(models.Model):
    """論文"""
    title = models.CharField('Title', max_length=200)
    abst = models.TextField('Abstract')

    def __tuple__(self):
        return self.title, self.abst
