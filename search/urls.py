from django.conf.urls import url
from search import views

urlpatterns = [
    # ToDoリスト
    # url(r'^ToDoList/$', views.ToDoList_list, name='todolist_list'),   # 一覧
    # url(r'^Paper/(?P<paper_id>\d+)/$', views.ToDoList_edit, name='paper'),  # 修正

    # 検索
    url(r'^index/$', views.Index, name='do_search'),
    # 内容
    url(r'^content/(?P<paper_id>\d+)/$', views.Content, name='content'),
]
