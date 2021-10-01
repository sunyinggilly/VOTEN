"""earthquake URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.urls import path
import sys
from . import COVTYPE_view, IJCAI_view, KDC_view, CENS_view
# from . import IJCAI_local_view

urlpatterns = [
    path('local/covtype', COVTYPE_view.local_init_page),
    path('global/covtype', COVTYPE_view.global_init_page),
    path('local/kdc', KDC_view.local_init_page),
    path('global/kdc', KDC_view.global_init_page),
    path('local/ijcai', IJCAI_view.local_init_page),
    path('global/ijcai', IJCAI_view.global_init_page),
    path('local/cens', CENS_view.local_init_page),
    path('global/cens', CENS_view.global_init_page)
]
