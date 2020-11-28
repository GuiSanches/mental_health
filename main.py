from utils import plot, granger_causuality, diff_primeira_ordem, centraliza_curva_covid, normaliza_lista,\
                  le_curva_covid_EUA, le_curva_mental_trends, le_curva_covid, le_curva_mental,\
                  calcula_media_periodo, substitui_media_periodo
from sys import argv

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

def computa_curvas(sigla: str, estado: str, doenca: str, estadual: bool):
    curva_covid, datas_covid = le_curva_covid(sigla) if estadual else le_curva_covid_EUA()
    curva_mental, datas_mental = le_curva_mental(estado, doenca)

    curva_covid, datas_covid = centraliza_curva_covid(datas_covid, curva_covid, inicio=datas_mental[0][0], fim=datas_mental[-1][1])

    curva_covid, datas_covid = substitui_media_periodo(curva_covid, datas_covid, datas_mental)

    datas_mental = [calcula_media_periodo(inicio, fim) for inicio, fim in datas_mental]

    curva_covid = normaliza_lista(curva_covid)

    return curva_covid, datas_covid, curva_mental, datas_mental

def google_trends(termo_pesquisa: str):
    curva_mental, datas_mental = le_curva_mental_trends(termo_pesquisa)

    curva_covid, datas_covid = le_curva_covid_EUA()

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
