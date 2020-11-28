from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import grangercausalitytests
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import numpy as np
import re, datetime
from sys import argv
from scipy.stats import spearmanr

atributo_depressao = 'Symptoms of Depressive Disorder'
atributo_ansiedade = 'Symptoms of Anxiety Disorder'
atributo_ambos = 'Symptoms of Anxiety Disorder or Depressive Disorder'

csv_covid = 'rows.csv accessType=DOWNLOAD.csv'
csv_covid_EUA = 'Dataset-OWID-covid.csv'
csv_mental = 'Indicators_of_Anxiety_or_Depression_Based_on_Reported_Frequency_of_Symptoms_During_Last_7_Days.csv'
csv_google_trends = 'google-data.csv'

estados = [
    ('AL', 'Alabama'),
    ('AK', 'Alaska'),
    ('AZ', 'Arizona'),
    ('AR', 'Arkansas'),
    ('CA', 'California'),
    ('CO', 'Colorado'),
    ('CT', 'Connecticut'),
    ('DE', 'Delaware'),
    ('FL', 'Florida'),
    ('GA', 'Georgia'),
    ('HI', 'Hawaii'),
    ('ID', 'Idaho'),
    ('IL', 'Illinois'),
    ('IN', 'Indiana'),
    ('IA', 'Iowa'),
    ('KS', 'Kansas'),
    ('KY', 'Kentucky'),
    ('LA', 'Louisiana'),
    ('ME', 'Maine'),
    ('MD', 'Maryland'),
    ('MA', 'Massachusetts'),
    ('MI', 'Michigan'),
    ('MN', 'Minnesota'),
    ('MS', 'Mississippi'),
    ('MO', 'Missouri'),
    ('MT', 'Montana'),
    ('NE', 'Nebraska'),
    ('NV', 'Nevada'),
    ('NH', 'New Hampshire'),
    ('NJ', 'New Jersey'),
    ('NM', 'New Mexico'),
    ('NY', 'New York'),
    ('NC', 'North Carolina'),
    ('ND', 'North Dakota'),
    ('OH', 'Ohio'),
    ('OK', 'Oklahoma'),
    ('OR', 'Oregon'),
    ('PA', 'Pennsylvania'),
    ('RI', 'Rhode Island'),
    ('SC', 'South Carolina'),
    ('SD', 'South Dakota'),
    ('TN', 'Tennessee'),
    ('TX', 'Texas'),
    ('UT', 'Utah'),
    ('VT', 'Vermont'),
    ('VA', 'Virginia'),
    ('WA', 'Washington'),
    ('WV', 'West Virginia'),
    ('WI', 'Wisconsin'),
    ('WY', 'Wyoming')
]

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

def normaliza_lista(lista):
    lista = [(float(i)-min(lista))/(max(lista)-min(lista)) for i in lista]
    return lista

# carrega o dataset de covid de determinado estado americano
def le_curva_covid(sigla):
    curva_covid = read_csv(csv_covid)

    curva_covid_estado = curva_covid[curva_covid['state'] == sigla]

    lista_curva_covid = list(curva_covid_estado['new_case'])
    lista_submission_date = list(curva_covid_estado['submission_date'])
    lista_data_covid = [get_data_covid(data) for data in lista_submission_date]

    return lista_curva_covid, lista_data_covid

# carrega o dataset de covid do país inteiro (EUA)
def le_curva_covid_EUA():
    curva_covid = read_csv(csv_covid_EUA)

    curva_covid_eua = curva_covid[curva_covid['location'] == 'United States']
    lista_curva_covid = list(curva_covid_eua['new_cases'])
    lista_submission_date = list(curva_covid_eua['date'])
    lista_data_covid = [get_data_covid_eua(data) for data in lista_submission_date]

    return lista_curva_covid, lista_data_covid

def le_curva_mental(estado, doenca):
    curva_mental = read_csv(csv_mental)
    curva_mental_filtered = curva_mental[curva_mental['Group'] == 'By State']

    atributo = eval('atributo_' + doenca)

    curva_mental_doenca = curva_mental_filtered[curva_mental_filtered['Indicator'] == atributo]

    curva_mental_estado = curva_mental_doenca[curva_mental_doenca['State'] == estado]

    lista_curva_mental = list(curva_mental_estado['Value'])
    lista_curva_mental_norm = normaliza_lista(lista_curva_mental)
    lista_time_period = list(curva_mental_estado['Time Period Label'])
    lista_datas_mental = [get_data(data) for data in lista_time_period]
    #lista_datas_medias_mental = [inicio + (fim - inicio)/2 for inicio, fim in lista_data_mental]

    return lista_curva_mental_norm, lista_datas_mental

# cortando todos os dados de covid anteriores e posteriores ao intervalo da curva de saúde mental
def centraliza_curva_covid(datas_covid, curva_covid, inicio, fim):
    inicio_covid = datas_covid.index(inicio)
    fim_covid = datas_covid.index(fim)
    lista_covid_sliced = curva_covid[inicio_covid:fim_covid+1]
    datas_covid_sliced = datas_covid[inicio_covid:fim_covid+1]
    return lista_covid_sliced, datas_covid_sliced

# reduzindo a quantidade de ocorrências de covid
# calcula a média de cada período do dataset de saúde mental e substitui por esse valor
def substitui_media_periodo(curva_covid, datas_covid, datas_mental):
    lista_covid_final = []
    lista_datas_covid_final = []
    for periodo in datas_mental:
        inicio, fim = periodo[0], periodo[1]
        media = calcula_media_periodo(inicio, fim)
        index_inicial = datas_covid.index(inicio)
        index_final = datas_covid.index(fim)
        valor_medio = np.mean(curva_covid[index_inicial:index_final+1])
        lista_covid_final.append(valor_medio)
        lista_datas_covid_final.append(media)
    return lista_covid_final, lista_datas_covid_final

def calcula_spline(curva_covid, datas_covid, curva_mental, datas_mental, n_samples):
    datas_covid, curva_covid = spline(datas_covid, curva_covid, n_samples)
    datas_mental, curva_mental = spline(datas_mental, curva_mental, n_samples)
    return curva_covid, datas_covid, curva_mental, datas_mental

def plot(curva_covid, datas_covid, curva_mental, datas_mental, label_mental):
    if datas_covid is None and datas_mental is None:
        plt.plot(curva_mental, label=label_mental)
        plt.plot(curva_covid, label='Casos covid')
    else:
        plt.plot(datas_mental, curva_mental, label=label_mental)
        plt.plot(datas_covid, curva_covid, label='Casos covid')
    plt.xlabel('Data da observação')
    plt.ylabel('Quantidade normalizada de casos')
    plt.legend(loc='upper left')
    plt.show()

def correlacao(curva_covid, curva_mental):
    pearson = np.corrcoef(curva_covid, curva_mental)[0][1]
    spearman = spearmanr(curva_covid, curva_mental)[0]
    return pearson, spearman

def granger_causuality(curva_covid, curva_mental, lag_max=4):
    df = DataFrame(columns=['mental', 'covid'], data=zip(curva_mental, curva_covid))
    return grangercausalitytests(df, lag_max)

def diff_primeira_ordem(curva_covid, curva_mental):
    return np.diff(curva_covid), np.diff(curva_mental)

def computa_curvas(sigla: str, estado: str, doenca: str, estadual: bool):
    curva_covid, datas_covid = le_curva_covid(sigla) if estadual else le_curva_covid_EUA()
    curva_mental, datas_mental = le_curva_mental(estado, doenca)

    curva_covid, datas_covid = centraliza_curva_covid(datas_covid, curva_covid, inicio=datas_mental[0][0], fim=datas_mental[-1][1])

    curva_covid, datas_covid = substitui_media_periodo(curva_covid, datas_covid, datas_mental)

    datas_mental = [calcula_media_periodo(inicio, fim) for inicio, fim in datas_mental]

    curva_covid = normaliza_lista(curva_covid)

    return curva_covid, datas_covid, curva_mental, datas_mental

def google_trends(termo_pesquisa: str):
    curva_mental = read_csv(csv_google_trends)
    curva_mental.columns = ['Dia', 'Depressão', 'Ansiedade', 'Suicídio', 'Insônia', 'Saúde mental']

    curva_covid, datas_covid = le_curva_covid_EUA()

    lista_time_period = list(curva_mental['Dia'])
    datas_mental = [get_data_covid_eua(data) for data in lista_time_period]

    curva_mental = curva_mental[termo_pesquisa]
    curva_covid = [i * 100 for i in normaliza_lista(curva_covid)]

    curva_covid, datas_covid = centraliza_curva_covid(datas_covid, curva_covid, inicio=datas_mental[0], fim=datas_mental[-1])

    return curva_covid, datas_covid, curva_mental, datas_mental

def testes_google_trends():
    curva_covid, datas_covid, curva_mental, datas_mental = google_trends(argv[1])
    print(granger_causuality(curva_covid, curva_mental), end="\n\n")
    plot(curva_covid, datas_covid, curva_mental, datas_mental, label_mental=('Pesquisas de ' + argv[1]))
    curva_covid, curva_mental = diff_primeira_ordem(curva_covid, curva_mental)
    print(granger_causuality(curva_covid, curva_mental), end="\n\n")
    plot(curva_covid, None, curva_mental, None, label_mental=('Pesquisas de ' + argv[1]))

def testes_cdc():
    curva_covid, datas_covid, curva_mental, datas_mental = computa_curvas(argv[2], argv[1], argv[3], estadual=True)
    print(granger_causuality(curva_covid, curva_mental), end="\n\n")
    plot(curva_covid, datas_covid, curva_mental, datas_mental, label_mental=('Casos de ' + argv[3]))
    curva_covid, curva_mental = diff_primeira_ordem(curva_covid, curva_mental)
    print(granger_causuality(curva_covid, curva_mental), end="\n\n")
    plot(curva_covid, None, curva_mental, None, label_mental=('Casos de ' + argv[3]))

testes_cdc()

#correlacoes = []
#for doenca in ['ansiedade', 'depressao', 'ambos']:
#    for sigla, estado in estados:
#        pearson, spearman = computa_curvas(sigla, estado, doenca, plot=False, estadual=True)
#        correlacoes.append((estado, doenca, pearson, spearman))
#
#correlacoes_ordenadas = sorted(correlacoes, key=lambda item : item[3], reverse=True)
#for estado, doenca, pearson, spearman in correlacoes_ordenadas:
#    print(estado, doenca, pearson, spearman, end='\n\n')
