"""auth URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
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
from django.conf.urls import url, include
from django.contrib import admin
from django.contrib.auth import views as auth_views

from core import views as core_views

urlpatterns = [
    #url(r'^$', core_views.index, name='index'),
    url(r'^$', auth_views.login, {'template_name': 'core/index_login.html'}, name='index'),
    url(r'^login/$', auth_views.login, {'template_name': 'core/index_login.html'}, name='login'),
    url(r'^admin/', admin.site.urls),
    url(r'^logout/$', auth_views.logout, {'next_page': 'index'}, name='logout'),
    url(r'^inregistrare/$', core_views.signup, name='inregistrare'),
    url(r'^barchart/$', core_views.barchart, name='barchart'),
    url(r'^piechart/$', core_views.piechart, name='piechart'),
    url(r'^home/$', core_views.home_view, name='home'),
]
