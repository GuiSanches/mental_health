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
csv_covid_EUA = 'owid-covid-data.csv'
csv_mental = 'Indicators_of_Anxiety_or_Depression_Based_on_Reported_Frequency_of_Symptoms_During_Last_7_Days(1).csv'

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
    lista_data_mental = [get_data(data) for data in lista_time_period]
    lista_datas_medias_mental = [inicio + (fim - inicio)/2 for inicio, fim in lista_data_mental]

    return lista_curva_mental_norm, lista_datas_medias_mental

# cortando todos os dados de covid anteriores e posteriores ao intervalo da curva de saúde mental
def centraliza_curva_covid(datas_covid, datas_mental, curva_covid):
    inicio = datas_covid.index(datas_mental[0][0])
    fim = datas_covid.index(datas_mental[-1][1])
    lista_covid_sliced = curva_covid[inicio:fim+1]
    datas_covid_sliced = datas_covid[inicio:fim+1]
    return lista_covid_sliced, datas_covid_sliced

# reduzindo a quantidade de ocorrências de covid
# calcula a média de cada período do dataset de saúde mental e substitui por esse valor
def calcula_substitui_media_periodo(curva_covid, datas_covid, datas_mental):
    lista_covid_final = []
    lista_datas_covid_final = []
    for periodo in datas_mental:
        inicio, fim = periodo[0], periodo[1]
        media = inicio + (fim - inicio)/2
        index_inicial = datas_covid.index(inicio)
        index_final = datas_covid.index(fim)
        valor_medio = np.mean(curva_covid[index_inicial:index_final+1])
        lista_covid_final.append(valor_medio)
        lista_datas_covid_final.append(media)
    return lista_covid_final, lista_datas_covid_final

def computa_curvas(sigla: str, estado: str, doenca: str, plot: bool, estadual: bool):
    curva_covid, datas_covid = le_curva_covid(sigla) if estadual else le_curva_covid_EUA()
    curva_mental, datas_mental = le_curva_mental(estado, doenca)

    curva_covid, datas_covid = centraliza_curva_covid(datas_covid, datas_mental, curva_covid)

    curva_covid, datas_covid = calcula_substitui_media_periodo(curva_covid, datas_covid, datas_mental)

    curva_covid = normaliza_lista(curva_covid)

    pearson = np.corrcoef(curva_covid, curva_mental)[0][1]
    spearman = spearmanr(curva_covid, curva_mental)[0]

    print("Pearson:", pearson)
    print("Spearman:", spearman)

    df = DataFrame(columns=['covid', 'mental'], data=zip(curva_covid, curva_mental))
    print(grangercausalitytests(df, 4), end="\n\n")

    if plot:
        plt.plot(datas_mental, curva_mental, label=('Casos de ' + doenca))
        plt.plot(datas_covid, curva_covid, label='Casos covid')
        plt.xlabel('Data da observação')
        plt.ylabel('Quantidade normalizada de casos')
        plt.legend(loc='upper left')
        plt.show()

    return pearson, spearman

def google_trends():
    curva_mental = read_csv('multiTimeline_3.csv')
    curva_mental.columns = ['Dia', 'Depressão', 'Ansiedade', 'Suicídio', 'Insônia', 'Saúde mental']

    curva_covid, datas_covid = le_curva_covid_EUA()

    lista_time_period = list(curva_mental['Dia'])
    lista_data_mental = [get_data_covid_eua(data) for data in lista_time_period]

    lista_covid_norm = [100*(float(i)-min(curva_covid))/(max(curva_covid)-min(curva_covid)) for i in curva_covid]

    for col in curva_mental.columns[1:]:
        #print(col)
        #print(np.corrcoef(lista_covid_norm, list(curva_mental[col])))
        #print(spearmanr(lista_covid_norm, list(curva_mental[col])))
        #print()
        plt.plot(lista_data_mental, list(curva_mental[col]), label=('Pesquisas de ' + col))

    plt.plot(datas_covid, lista_covid_norm, label='Casos covid')
    plt.xlabel('Data da observação')
    plt.ylabel('Quantidade normalizada de casos')
    plt.legend(loc='upper left')
    plt.show()

#google_trends()

computa_curvas(argv[2], argv[1], argv[3], plot=True, estadual=True)

#correlacoes = []
#for doenca in ['ansiedade', 'depressao', 'ambos']:
#    for sigla, estado in estados:
#        pearson, spearman = computa_curvas(sigla, estado, doenca, plot=False, estadual=True)
#        correlacoes.append((estado, doenca, pearson, spearman))
#
#correlacoes_ordenadas = sorted(correlacoes, key=lambda item : item[3], reverse=True)
#for estado, doenca, pearson, spearman in correlacoes_ordenadas:
#    print(estado, doenca, pearson, spearman, end='\n\n')
