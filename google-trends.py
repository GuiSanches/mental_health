#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 18:18:13 2020

@author: guilherme
"""
from pandas import read_csv
import numpy as np
import re, datetime
import sys

#Visualization
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.colors import ListedColormap
from matplotlib import cm
import seaborn as sns
import matplotlib.patches as mpatches

#Plotly Libraris
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot, plot
from plotly import tools
from IPython.display import display, HTML, display_html

def get_data_covid_eua(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d')

def google_trends():
    curva_mental = read_csv('google-data.csv', index_col='Dia')
    curva_mental.columns = ['Depressão', 'Ansiedade', 'Suicídio', 'Insônia', 'Saúde mental']

    curva_covid = read_csv('owid-covid-data.csv')

    lista_time_period = list(curva_mental.index.values)
    lista_data_mental = [get_data_covid_eua(data) for data in lista_time_period]

    curva_covid_eua = curva_covid[curva_covid['location'] == 'United States']
    lista_curva_covid = list(curva_covid_eua['new_cases'])
    lista_submission_date = list(curva_covid_eua['date'])
    lista_data_covid = [get_data_covid_eua(data) for data in lista_submission_date]
#
    lista_covid_norm = [100*(float(i)-min(lista_curva_covid))/(max(lista_curva_covid)-min(lista_curva_covid)) for i in lista_curva_covid]
    
    fig = go.Figure()
    for col in curva_mental.columns:
        fig.add_trace(go.Scatter(
        x=lista_data_mental, 
        y=list(curva_mental[col]),
        name='Pesquisas de ' + col,
        mode='lines+markers'))
        #plt.plot(lista_data_mental, list(curva_mental[col]), label=('Pesquisas de ' + col))
        
    fig.add_trace(go.Scatter(
        x=lista_data_covid, 
        y=lista_covid_norm,
        name='Casos covid',
        mode='lines+markers'))
        
    fig.update_layout(
        title="Pesquisas durante pandemia Covid-19",
        xaxis_title="'Data da observação'",
        yaxis_title="Quantidade normalizada de casos",
        xaxis=dict(type='category')
  )
    fig.show()
    plot(fig)
    print('run')
    
google_trends()

