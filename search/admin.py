from django.contrib import admin
from search.models import Paper


class PaperAdmin(admin.ModelAdmin):
    list_display = ('id', 'title','abst')  # 一覧に出したい項目
    list_display_links = ('id', 'title','abst')  # 修正リンクでクリックできる項目
admin.site.register(Paper, PaperAdmin)
