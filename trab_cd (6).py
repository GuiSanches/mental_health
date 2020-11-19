from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import re, datetime
import sys
from scipy.stats import spearmanr

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

def computa_estados(sigla, estado, doenca):
    curva_mental = read_csv('Indicators_of_Anxiety_or_Depression_Based_on_Reported_Frequency_of_Symptoms_During_Last_7_Days(1).csv')

    curva_mental_filtered = curva_mental[curva_mental['Group'] == 'By State']

    curva_mental_depressao = curva_mental_filtered[curva_mental_filtered['Indicator'] == 'Symptoms of Depressive Disorder']
    curva_mental_ansiedade = curva_mental_filtered[curva_mental_filtered['Indicator'] == 'Symptoms of Anxiety Disorder']
    curva_mental_ambos = curva_mental_filtered[curva_mental_filtered['Indicator'] == 'Symptoms of Anxiety Disorder or Depressive Disorder']

    curva_covid = read_csv('rows.csv accessType=DOWNLOAD.csv')

    curva_mental_estado = eval('curva_mental_' + doenca)[eval('curva_mental_' + doenca)['State'] == estado]
    lista_curva_mental = list(curva_mental_estado['Value'])
    lista_curva_mental_norm = [(float(i)-min(lista_curva_mental))/(max(lista_curva_mental)-min(lista_curva_mental)) for i in lista_curva_mental]
    lista_time_period = list(curva_mental_estado['Time Period Label'])
    lista_data_mental = [get_data(data) for data in lista_time_period]
    lista_datas_medias_mental = [inicio + (fim - inicio)/2 for inicio, fim in lista_data_mental]

    curva_covid_estado = curva_covid[curva_covid['state'] == sigla]
    lista_curva_covid = list(curva_covid_estado['new_case'])
    lista_submission_date = list(curva_covid_estado['submission_date'])
    lista_data_covid = [get_data_covid(data) for data in lista_submission_date]

    # cortando todos os dados de covid anteriores e posteriores ao intervalo da curva de saúde mental
    inicio = lista_data_covid.index(lista_data_mental[0][0])
    fim = lista_data_covid.index(lista_data_mental[-1][1])
    lista_covid_sliced = lista_curva_covid[inicio:fim+1]
    lista_data_covid_sliced = lista_data_covid[inicio:fim+1]

    # reduzindo a quantidade de ocorrências de covid
    # calcula a média de cada período do dataset de saúde mental
    lista_covid_final = []
    lista_datas_covid_final = []
    for periodo in lista_data_mental:
        inicio, fim = periodo[0], periodo[1]
        media = inicio + (fim - inicio)/2
        index_inicial = lista_data_covid_sliced.index(inicio)
        index_final = lista_data_covid_sliced.index(fim)
        valor_medio = np.mean(lista_covid_sliced[index_inicial:index_final+1])
        lista_covid_final.append(valor_medio)
        lista_datas_covid_final.append(media)

    lista_covid_final = [(float(i)-min(lista_covid_final))/(max(lista_covid_final)-min(lista_covid_final)) for i in lista_covid_final]

    pearson = np.corrcoef(lista_covid_final, lista_curva_mental_norm)[0][1]
    spearman = spearmanr(lista_covid_final, lista_curva_mental_norm)[0]

    #print("Pearson:", pearson)
    #print("Spearman:", spearman)

    return pearson, spearman

def computa_estados_argv():
    curva_mental = read_csv('Indicators_of_Anxiety_or_Depression_Based_on_Reported_Frequency_of_Symptoms_During_Last_7_Days(1).csv')

    curva_mental_filtered = curva_mental[curva_mental['Group'] == 'By State']

    curva_mental_depressao = curva_mental_filtered[curva_mental_filtered['Indicator'] == 'Symptoms of Depressive Disorder']
    curva_mental_ansiedade = curva_mental_filtered[curva_mental_filtered['Indicator'] == 'Symptoms of Anxiety Disorder']
    curva_mental_ambos = curva_mental_filtered[curva_mental_filtered['Indicator'] == 'Symptoms of Anxiety Disorder or Depressive Disorder']

    curva_covid = read_csv('rows.csv accessType=DOWNLOAD.csv')

    curva_mental_estado = eval('curva_mental_' + sys.argv[3])[eval('curva_mental_' + sys.argv[3])['State'] == sys.argv[1]]
    lista_curva_mental = list(curva_mental_estado['Value'])
    lista_curva_mental_norm = [(float(i)-min(lista_curva_mental))/(max(lista_curva_mental)-min(lista_curva_mental)) for i in lista_curva_mental]
    lista_time_period = list(curva_mental_estado['Time Period Label'])
    lista_data_mental = [get_data(data) for data in lista_time_period]
    lista_datas_medias_mental = [inicio + (fim - inicio)/2 for inicio, fim in lista_data_mental]

    curva_covid_estado = curva_covid[curva_covid['state'] == sys.argv[2]]
    lista_curva_covid = list(curva_covid_estado['new_case'])
    lista_submission_date = list(curva_covid_estado['submission_date'])
    lista_data_covid = [get_data_covid(data) for data in lista_submission_date]

    # cortando todos os dados de covid anteriores e posteriores ao intervalo da curva de saúde mental
    inicio = lista_data_covid.index(lista_data_mental[0][0])
    fim = lista_data_covid.index(lista_data_mental[-1][1])
    lista_covid_sliced = lista_curva_covid[inicio:fim+1]
    lista_data_covid_sliced = lista_data_covid[inicio:fim+1]

    # reduzindo a quantidade de ocorrências de covid
    # calcula a média de cada período do dataset de saúde mental
    lista_covid_final = []
    lista_datas_covid_final = []
    for periodo in lista_data_mental:
        inicio, fim = periodo[0], periodo[1]
        media = inicio + (fim - inicio)/2
        index_inicial = lista_data_covid_sliced.index(inicio)
        index_final = lista_data_covid_sliced.index(fim)
        valor_medio = np.mean(lista_covid_sliced[index_inicial:index_final+1])
        lista_covid_final.append(valor_medio)
        lista_datas_covid_final.append(media)

    lista_covid_final = [(float(i)-min(lista_covid_final))/(max(lista_covid_final)-min(lista_covid_final)) for i in lista_covid_final]

    print(np.corrcoef(lista_covid_final, lista_curva_mental_norm))
    print(spearmanr(lista_covid_final, lista_curva_mental_norm))

    plt.plot(lista_datas_medias_mental, lista_curva_mental_norm, label=('Casos de ' + sys.argv[3]))
    plt.plot(lista_datas_covid_final, lista_covid_final, label='Casos covid')
    plt.xlabel('Data da observação')
    plt.ylabel('Quantidade normalizada de casos')
    plt.legend(loc='upper left')
    plt.show()

def computa_EUA():
    curva_mental = read_csv('Indicators_of_Anxiety_or_Depression_Based_on_Reported_Frequency_of_Symptoms_During_Last_7_Days(1).csv')

    curva_mental_filtered = curva_mental[curva_mental['Group'] == 'By State']

    curva_mental_depressao = curva_mental_filtered[curva_mental_filtered['Indicator'] == 'Symptoms of Depressive Disorder']
    curva_mental_ansiedade = curva_mental_filtered[curva_mental_filtered['Indicator'] == 'Symptoms of Anxiety Disorder']
    curva_mental_ambos = curva_mental_filtered[curva_mental_filtered['Indicator'] == 'Symptoms of Anxiety Disorder or Depressive Disorder']

    curva_covid = read_csv('owid-covid-data.csv')

    curva_mental_estado = eval('curva_mental_' + sys.argv[3])[eval('curva_mental_' + sys.argv[3])['State'] == sys.argv[1]]
    lista_curva_mental = list(curva_mental_estado['Value'])
    lista_curva_mental_norm = [(float(i)-min(lista_curva_mental))/(max(lista_curva_mental)-min(lista_curva_mental)) for i in lista_curva_mental]
    lista_time_period = list(curva_mental_estado['Time Period Label'])
    lista_data_mental = [get_data(data) for data in lista_time_period]
    lista_datas_medias_mental = [inicio + (fim - inicio)/2 for inicio, fim in lista_data_mental]

    curva_covid_eua = curva_covid[curva_covid['location'] == 'United States']
    lista_curva_covid = list(curva_covid_eua['new_cases'])
    lista_submission_date = list(curva_covid_eua['date'])
    lista_data_covid = [get_data_covid_eua(data) for data in lista_submission_date]

    # cortando todos os dados de covid anteriores e posteriores ao intervalo da curva de saúde mental
    inicio = lista_data_covid.index(lista_data_mental[0][0])
    fim = lista_data_covid.index(lista_data_mental[-1][1])
    lista_covid_sliced = lista_curva_covid[inicio:fim+1]
    lista_data_covid_sliced = lista_data_covid[inicio:fim+1]

    # reduzindo a quantidade de ocorrências de covid
    # calcula a média de cada período do dataset de saúde mental
    lista_covid_final = []
    lista_datas_covid_final = []
    for periodo in lista_data_mental:
        inicio, fim = periodo[0], periodo[1]
        media = inicio + (fim - inicio)/2
        index_inicial = lista_data_covid_sliced.index(inicio)
        index_final = lista_data_covid_sliced.index(fim)
        valor_medio = np.mean(lista_covid_sliced[index_inicial:index_final+1])
        lista_covid_final.append(valor_medio)
        lista_datas_covid_final.append(media)

    lista_covid_final = [(float(i)-min(lista_covid_final))/(max(lista_covid_final)-min(lista_covid_final)) for i in lista_covid_final]

    print(np.corrcoef(lista_covid_final, lista_curva_mental_norm))
    print(spearmanr(lista_covid_final, lista_curva_mental_norm))

    plt.plot(lista_datas_medias_mental, lista_curva_mental_norm, label=('Casos de ' + sys.argv[3]))
    plt.plot(lista_datas_covid_final, lista_covid_final, label='Casos covid')
    plt.xlabel('Data da observação')
    plt.ylabel('Quantidade normalizada de casos')
    plt.legend(loc='upper left')
    plt.show()

def google_trends():
    curva_mental = read_csv('multiTimeline_3.csv')
    curva_mental.columns = ['Dia', 'Depressão', 'Ansiedade', 'Suicídio', 'Insônia', 'Saúde mental']

    curva_covid = read_csv('owid-covid-data.csv')

    lista_time_period = list(curva_mental['Dia'])
    lista_data_mental = [get_data_covid_eua(data) for data in lista_time_period]

    curva_covid_eua = curva_covid[curva_covid['location'] == 'United States']
    lista_curva_covid = list(curva_covid_eua['new_cases'])
    lista_submission_date = list(curva_covid_eua['date'])
    lista_data_covid = [get_data_covid_eua(data) for data in lista_submission_date]

#    # cortando todos os dados de covid anteriores e posteriores ao intervalo da curva de saúde mental
#    inicio = lista_data_covid.index(lista_data_mental[0][0])
#    fim = lista_data_covid.index(lista_data_mental[-1][1])
#    lista_covid_sliced = lista_curva_covid[inicio:fim+1]
#    lista_data_covid_sliced = lista_data_covid[inicio:fim+1]
#
#    # reduzindo a quantidade de ocorrências de covid
#    # calcula a média de cada período do dataset de saúde mental
#    lista_covid_final = []
#    lista_datas_covid_final = []
#    for periodo in lista_data_mental:
#        inicio, fim = periodo[0], periodo[1]
#        media = inicio + (fim - inicio)/2
#        index_inicial = lista_data_covid_sliced.index(inicio)
#        index_final = lista_data_covid_sliced.index(fim)
#        valor_medio = np.mean(lista_covid_sliced[index_inicial:index_final+1])
#        lista_covid_final.append(valor_medio)
#        lista_datas_covid_final.append(media)
#
    lista_covid_norm = [100*(float(i)-min(lista_curva_covid))/(max(lista_curva_covid)-min(lista_curva_covid)) for i in lista_curva_covid]

    for col in curva_mental.columns[1:]:
        #print(col)
        #print(np.corrcoef(lista_covid_norm, list(curva_mental[col])))
        #print(spearmanr(lista_covid_norm, list(curva_mental[col])))
        #print()
        plt.plot(lista_data_mental, list(curva_mental[col]), label=('Pesquisas de ' + col))

    plt.plot(lista_data_covid, lista_covid_norm, label='Casos covid')
    plt.xlabel('Data da observação')
    plt.ylabel('Quantidade normalizada de casos')
    plt.legend(loc='upper left')
    plt.show()

google_trends()

#computa_estados_argv()

#correlacoes = []
#for doenca in ['ansiedade', 'depressao', 'ambos']:
#    for sigla, estado in estados:
#        pearson, spearman = computa_estados(sigla, estado, doenca)
#        correlacoes.append((estado, doenca, pearson, spearman))
#
#correlacoes_ordenadas = sorted(correlacoes, key=lambda item : item[3], reverse=True)
#for estado, doenca, pearson, spearman in correlacoes_ordenadas:
#    print(estado, doenca, pearson, spearman, end='\n\n')
