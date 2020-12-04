#from utils import plot, granger_causuality, diff_primeira_ordem, centraliza_curva_covid, normaliza_dataframe,\
#                  le_curva_covid_EUA, le_curva_mental_trends, le_curva_covid, le_curva_mental,\
#                  calcula_media_periodo, substitui_media_periodo

from utils import *
from sys import argv
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

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
    df_covid = le_curva_covid(sigla) if estadual else le_curva_covid_EUA(normaliza=False)
    df_mental, periodos_mental = le_curva_mental(estado, doenca) if estadual else le_curva_mental_EUA(doenca)

    df_covid = centraliza_curva_covid(df_covid, inicio=df_mental.index[0], fim=df_mental.index[-1])

    df_covid = substitui_media_periodo(df_covid, periodos_mental)

    df_covid = normaliza_dataframe(df_covid)

    return df_covid, df_mental

def google_trends(termo_pesquisa: str):
    df_mental = le_curva_mental_trends(termo_pesquisa)
    df_covid = le_curva_covid_EUA()

    df_covid = centraliza_curva_covid(df_covid, inicio=df_mental.index[0], fim=df_mental.index[-1])

    return df_covid, df_mental

def testes_google_trends():
    df_covid, df_mental = google_trends(argv[1])
    #print(granger_causuality(df_covid, df_mental), end="\n\n")
    plot(df_covid, df_mental, label_mental=('Pesquisas de ' + argv[1]))
    df_covid = diff_primeira_ordem(df_covid)[1:]
    df_mental = diff_primeira_ordem(df_mental)[1:]
    #print(granger_causuality(df_covid, df_mental), end="\n\n")
    plot(df_covid, df_mental, label_mental=('Pesquisas de ' + argv[1]))
    plot_pacf(df_mental)
    plt.show()
    for lag in range(1, 40):
        mental_series = df_mental[argv[1]].iloc[lag:]
        covid_series = df_covid['new_cases'].iloc[:-lag]
        print('Lag:', lag)
        print(pearsonr(mental_series, covid_series))
        print('------')
    df = concat([df_mental, df_covid], axis=1)
    model = VAR(df)
    model_fit = model.fit(maxlags=35)
    print(model_fit.summary())
    return df_covid, df_mental

def previsao_ansiedade(df_covid, df_mental, inicio):
    coefs = [-0.547341, -0.580165, -0.438828, -0.453479, -0.335704, -0.340717, -0.240634, -0.255337, 0.848681, -0.292593, 0.834970, -0.251307, -0.264414, -0.310971, -0.233544, -0.276718, -0.837788, -0.269932, -0.845041, -0.929429, -1.030544]
    lags = [1, 2, 3, 4, 5, 6, 10, 12, 12, 13, 13, 14, 15, 16, 17, 18, 18, 19, 19, 20, 21]
    curvas = ['mental', 'mental', 'mental', 'mental', 'mental', 'mental', 'mental', 'mental', 'covid', 'mental', 'covid', 'mental', 'mental', 'mental', 'mental', 'mental', 'covid', 'mental', 'covid', 'covid', 'covid']
    df_ansiedade = DataFrame(columns=df_mental.columns)
    for i, data in enumerate(df_mental.index[inicio:]):
        parcial_sum = 0
        for coef, lag, curva in zip(coefs, lags, curvas):
            parcial_sum += coef * eval('df_'+curva).iloc[i-lag][0]
        df_ansiedade.loc[data] = parcial_sum
    plt.plot(df_mental, label="original")
    plt.plot(df_ansiedade, label="previsão")
    plt.legend(loc='upper left')
    plt.show()

def testes_cdc():
    df_covid, df_mental = computa_curvas(argv[2], argv[1], argv[3], estadual=False)

    print(granger_causuality(df_covid, df_mental), end="\n\n")
    #plot(df_covid, df_mental, label_mental=('Casos de ' + argv[3]))
    df = concat([df_mental, df_covid], axis=1)
    model = VAR(df)
    model_fit = model.fit(maxlags=10)
    print(model_fit.summary())

    df_covid = diff_primeira_ordem(df_covid)[1:]
    df_mental = diff_primeira_ordem(df_mental)[1:]
    print(granger_causuality(df_covid, df_mental), end="\n\n")
    #plot(df_covid, df_mental, label_mental=('Casos de ' + argv[3]))
    df = concat([df_mental, df_covid], axis=1)
    model = VAR(df)
    model_fit = model.fit(maxlags=10)
    print(model_fit.summary())

    df_covid = diff_primeira_ordem(df_covid)[1:]
    df_mental = diff_primeira_ordem(df_mental)[1:]
    print(granger_causuality(df_covid, df_mental), end="\n\n")
    #plot(df_covid, df_mental, label_mental=('Casos de ' + argv[3]))
    df = concat([df_mental, df_covid], axis=1)
    model = VAR(df)
    model_fit = model.fit(maxlags=15)
    print(model_fit.summary())

    return df_covid, df_mental

df_covid, df_mental = testes_google_trends()
previsao_ansiedade(df_covid, df_mental, 22)

#correlacoes = []
#for doenca in ['ansiedade', 'depressao', 'ambos']:
#    for sigla, estado in estados:
#        pearson, spearman = computa_curvas(sigla, estado, doenca, plot=False, estadual=True)
#        correlacoes.append((estado, doenca, pearson, spearman))
#
#correlacoes_ordenadas = sorted(correlacoes, key=lambda item : item[3], reverse=True)
#for estado, doenca, pearson, spearman in correlacoes_ordenadas:
#    print(estado, doenca, pearson, spearman, end='\n\n')
