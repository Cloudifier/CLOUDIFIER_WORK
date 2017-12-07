from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import SignUpForm
import sqlite3
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.offline import plot
import os

@login_required
def home_view(request):
    if request.method == 'GET':
        return render(request, 'core/mainpage_get.html')

    if request.method == 'POST':
        df = None
        conn = sqlite3.connect('mysqlite3.db')
        c = conn.cursor()

        names = []
        date_entered = []
        industries = []
        billing_address_country = []
        cmd = "SELECT name, date_entered, industry, billing_address_country FROM ACCOUNTS  WHERE LENGTH(industry)>1"
        for crt_row in c.execute(cmd):
            names.append(crt_row[0])
            date_entered.append(crt_row[1])
            industries.append(crt_row[2])
            billing_address_country.append(crt_row[3])

        aux_dict = {'name': names, 'date_entered': date_entered, 'industry': industries, 'billing_address_country': billing_address_country}
        df = pd.DataFrame(aux_dict)
        print(list(df.columns))

        if request.POST['searchField'] != '':
            df = df.loc[df['industry'] == request.POST['searchField']]

        if not df is None:
            return render(request, 'core/mainpage_post.html', {'df': df})
        else:
            return HttpResponse("Internal error")

df = None

@login_required
def barchart(request):
    conn = sqlite3.connect('mysqlite3.db')
    c = conn.cursor()
    
    industries = []
    cmd = "SELECT industry, COUNT(industry) FROM ACCOUNTS  WHERE LENGTH(industry)>1 GROUP BY industry"
    counts = []
    for crt_row in c.execute(cmd):
        industries.append(crt_row[0])
        counts.append(crt_row[1])

    labels = industries
    values = counts

    trace1 = go.Bar(x=labels, y=values, marker=dict(
            color='rgb(34,196,234))',
            line=dict(
                color='rgb(0,188,255)',
                width=1.5,
            )
        ),
        opacity=0.6)
    data = [trace1]
    layout = go.Layout(
        xaxis=dict(
            tickangle=-45,
            title='Industrie',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title='Numar companii',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ))
    fig = go.Figure(data=data, layout=layout)

    filename = os.path.dirname(os.path.abspath(__file__))
    templates_dir_name = os.path.join(filename, "templates/core/")
    filename = os.path.join(templates_dir_name, "barchart.html")
    plot(fig, config=dict(displayModeBar=False, showLink=False), filename=filename, auto_open=False)

    return render(request, 'core/barchart.html')

@login_required
def piechart(request):
    conn = sqlite3.connect('mysqlite3.db')
    c = conn.cursor()

    industries = []
    cmd = "SELECT industry, COUNT(industry) FROM ACCOUNTS  WHERE LENGTH(industry)>1 GROUP BY industry"
    counts = []
    for crt_row in c.execute(cmd):
        industries.append(crt_row[0])
        counts.append(crt_row[1])

    labels = industries
    values = counts

    trace1 = go.Pie(labels=labels, values=values)
    data = [trace1]
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)

    filename = os.path.dirname(os.path.abspath(__file__))
    templates_dir_name = os.path.join(filename, "templates/core/")
    filename = os.path.join(templates_dir_name, "piechart.html")
    plot(fig, config=dict(displayModeBar=False, showLink=False), filename=filename, auto_open=False)

    return render(request, 'core/piechart.html')


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('home')
    else:
        form = SignUpForm()
    return render(request, 'core/signup.html', {'form': form})
