from django.http import HttpResponse, HttpResponseRedirect
from .utils import CarsHelper
import json
from django.shortcuts import render
import pandas as pd
from .forms import SearchForm

def index(request):
    return render(request, 'api/index.html')

def api_view(request):
    if request.method == 'GET':
        return HttpResponse('Please make a POST request')

    if request.method == 'POST':
        if request.META['CONTENT_TYPE'] == 'application/json':
            helper = CarsHelper()
            response = helper.get_cars(request)
            return HttpResponse(json.dumps(response), content_type="application/json")
        response = {}
        response['responses'] = []
        response['responses'].append({'status': 'BAD_REQUEST', 'status_code': '400', 'description': 'Please send a JSON object'})
        return HttpResponse(json.dumps(response), content_type="application/json")

"""
def rawdata_view(request):
    df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
    #print(df.head())
    #return render(request, 'api/rawdata.html', {'df': df, 'rng': [i for i in range(5)]})
    template = 'api/rawdata.html'

    entries = [tuple(x) for x in df.to_records(index=False)]
    context = {'entries': entries}
    return render(request, template, context)
"""

df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")

def rawdata_view(request, template='api/rawdata.html', page_template='api/rawdata_page.html'):
    global df
    form = None
    if request.method == 'POST':
        form = SearchForm(request.POST)

    if form is not None:
        search_parameters = None
        if form.is_valid():
            search_parameters = (form.cleaned_data['CarID'], form.cleaned_data['Code'])

        if search_parameters is not None:
            if search_parameters[1] is not '':
                df = df.loc[df['species']==search_parameters[1]]

    else:
        page = request.GET.get('page', False)
        if page is False:
            df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")

    context = {
        'entry_list': [tuple(x) for x in df.to_records(index=False)],
        'page_template': page_template,
    }
    if request.is_ajax():
        template = page_template
    return render(request, template, context)

"""
{% load el_pagination_tags %}

{% paginate entries %} {% get_pages %}
{% for entry in entries %}
    <h4>{{ entry.4 }}</h4>      
{% endfor %}
Showing entries
{{ pages.current_start_index }}-{{ pages.current_end_index }} of
{{ pages.total_count }}.
<br>
{{ pages }}
"""