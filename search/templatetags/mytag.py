from django import template
import numpy as np

register = template.Library()

@register.filter(name='show_list')
def show_list(tmp_list,index):
    return tmp_list[index]

@register.filter(name='range_make')
def range_make(num):
    num_list = np.arange(num)
    num_list = list(num_list)
    return num_list
