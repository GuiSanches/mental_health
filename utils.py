import datetime
import re

import matplotlib.pyplot as plt
import numpy as np

from pandas import DataFrame, read_csv, concat
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests

atributo_depressao = 'Symptoms of Depressive Disorder'
atributo_ansiedade = 'Symptoms of Anxiety Disorder'
atributo_ambos = 'Symptoms of Anxiety Disorder or Depressive Disorder'

csv_covid = 'rows.csv accessType=DOWNLOAD.csv'
csv_covid_EUA = 'owid-covid-data (1).csv'
csv_mental = 'Indicators_of_Anxiety_or_Depression_Based_on_Reported_Frequency_of_Symptoms_During_Last_7_Days.csv'
csv_google_trends = 'google-data.csv'

#recebe arrays x e y
def spline(x_axis, y_axis, n_samples):
    x_axis = [time.timestamp() for time in x_axis]
    f2 = interp1d(x_axis, y_axis, kind='cubic')
    xnew = np.linspace(x_axis[0], x_axis[-1], num=n_samples, endpoint=True)
    ynew = f2(xnew)
    xnew = [datetime.datetime.fromtimestamp(timestamp) for timestamp in xnew]
    return xnew, ynew

def calcula_media_periodo(inicio, fim):
    return inicio + (fim - inicio)/2

def get_data(date):
    date_regex = re.compile('([A-Z]{3,4} \d{1,2})', flags=re.MULTILINE | re.IGNORECASE)
    dates = date_regex.findall(date)
    begin = datetime.datetime.strptime(dates[0] + ' 2020', '%b %d %Y') if dates[0][3] == ' ' \
                                       else datetime.datetime.strptime(dates[0] + ' 2020', '%B %d %Y')
    end = datetime.datetime.strptime(dates[1] + ' 2020', '%b %d %Y') if dates[1][3] == ' ' \
                                     else datetime.datetime.strptime(dates[1] + ' 2020', '%B %d %Y')
    return begin, end

def get_data_covid(date):
    return datetime.datetime.strptime(date, '%m/%d/%Y')

def get_data_covid_eua(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d')

def normaliza_dataframe(df: DataFrame):
    media = df.mean()[0]
    desvio = df.std()[0]
    df = (df - media) / desvio
    return df

# carrega o dataset de covid de determinado estado americano
def le_curva_covid(sigla):
    curva_covid = read_csv(csv_covid)

    curva_covid_estado = curva_covid[curva_covid['state'] == sigla]

    df_covid = curva_covid_estado[['submission_date', 'new_case']]

    for index, data in enumerate(df_covid['submission_date']):
        df_covid['submission_date'].iloc[index] = get_data_covid(data)

    df_covid = df_covid.set_index('submission_date')

    return df_covid

# carrega o dataset de covid do país inteiro (EUA)
def le_curva_covid_EUA(normaliza=True):
    curva_covid = read_csv(csv_covid_EUA)

    curva_covid_eua = curva_covid[curva_covid['location'] == 'United States']

    df_covid = curva_covid_eua[['date', 'new_cases']]

    for index, data in enumerate(df_covid['date']):
        df_covid['date'].iloc[index] = get_data_covid_eua(data)

    df_covid = df_covid.set_index('date')

    if normaliza:
        df_covid = normaliza_dataframe(df_covid)

    return df_covid

def le_curva_mental(estado, doenca):
    curva_mental = read_csv(csv_mental)
    curva_mental_filtered = curva_mental[curva_mental['Group'] == 'By State']

    atributo = eval('atributo_' + doenca)

    curva_mental_doenca = curva_mental_filtered[curva_mental_filtered['Indicator'] == atributo]

    curva_mental_estado = curva_mental_doenca[curva_mental_doenca['State'] == estado]

    df_mental = curva_mental_estado[['Time Period Label', 'Value']]

    for index, data in enumerate(df_mental['Time Period Label']):
        df_mental['Time Period Label'].iloc[index] = get_data(data)

    periodos = df_mental['Time Period Label'].values

    df_mental['Time Period Label'] = [calcula_media_periodo(inicio, fim) for inicio, fim in periodos]

    df_mental = df_mental.set_index('Time Period Label')

    df_mental = normaliza_dataframe(df_mental)

    return df_mental, periodos

def le_curva_mental_EUA(doenca):
    curva_mental = read_csv(csv_mental)
    curva_mental_filtered = curva_mental[curva_mental['Group'] == 'National Estimate']

    atributo = eval('atributo_' + doenca)

    curva_mental_doenca = curva_mental_filtered[curva_mental_filtered['Indicator'] == atributo]

    curva_mental_valid = curva_mental_doenca[curva_mental_doenca['Phase'] != -1]

    df_mental = curva_mental_valid[['Time Period Label', 'Value']]

    for index, data in enumerate(df_mental['Time Period Label']):
        df_mental['Time Period Label'].iloc[index] = get_data(data)

    periodos = df_mental['Time Period Label'].values

    df_mental['Time Period Label'] = [calcula_media_periodo(inicio, fim) for inicio, fim in periodos]

    df_mental = df_mental.set_index('Time Period Label')

    df_mental = normaliza_dataframe(df_mental)

    return df_mental, periodos

def le_curva_mental_trends(termo_pesquisa: str):
    curva_mental = read_csv(csv_google_trends)
    curva_mental.columns = ['Dia', 'Depressão', 'Ansiedade', 'Suicídio', 'Insônia', 'Saúde mental']

    df_mental = curva_mental[['Dia', termo_pesquisa]]

    for index, data in enumerate(df_mental['Dia']):
        df_mental['Dia'].iloc[index] = get_data_covid_eua(data)

    df_mental = df_mental.set_index('Dia')

    df_mental = normaliza_dataframe(df_mental)

    return df_mental

# cortando todos os dados de covid anteriores e posteriores ao intervalo da curva de saúde mental
def centraliza_curva_covid(df_covid, inicio, fim):
    return df_covid.loc[inicio:fim]

# reduzindo a quantidade de ocorrências de covid
# calcula a média de cada período do dataset de saúde mental e substitui por esse valor
def substitui_media_periodo(df_covid, periodos_mental):
    df_covid_final = DataFrame(columns=df_covid.columns)
    for periodo in periodos_mental:
        inicio, fim = periodo[0], periodo[1]
        media = calcula_media_periodo(inicio, fim)
        df_covid_final.loc[media] = df_covid.loc[inicio:fim].mean()[0]
    return df_covid_final

def calcula_spline(curva_covid, datas_covid, curva_mental, datas_mental, n_samples):
    datas_covid, curva_covid = spline(datas_covid, curva_covid, n_samples)
    datas_mental, curva_mental = spline(datas_mental, curva_mental, n_samples)
    return curva_covid, datas_covid, curva_mental, datas_mental

def plot(df_covid, df_mental, label_mental):
    plt.plot(df_mental, label=label_mental)
    plt.plot(df_covid, label='Casos covid')

    plt.xlabel('Data da observação')
    plt.ylabel('Quantidade normalizada de casos')
    plt.legend(loc='upper left')

    plt.show()

def correlacao(curva_covid, curva_mental):
    pearson = np.corrcoef(curva_covid, curva_mental)[0][1]
    spearman = spearmanr(curva_covid, curva_mental)[0]
    return pearson, spearman

def granger_causuality(df_covid, df_mental, lag_max=4):
    df = concat([df_mental, df_covid], axis=1)
    return grangercausalitytests(df, lag_max)

def diff_primeira_ordem(df: DataFrame):
    return df.diff()
