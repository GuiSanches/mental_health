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
    #plot_pacf(df_mental)
    #plt.show()
    #for lag in range(1, 30):
    #    mental_series = df_mental[argv[1]].iloc[lag:]
    #    covid_series = df_covid['new_cases'].iloc[:-lag]
    #    print('Lag:', lag)
    #    print(pearsonr(mental_series, covid_series))
    #    print('------')
    #df = concat([df_mental, df_covid], axis=1)
    #model = VAR(df)
    #model_fit = model.fit(maxlags=8)
    #print(model_fit.summary())
    plot(df_covid, df_mental, label_mental=('Pesquisas de ' + argv[1]))

    return df_covid, df_mental

def find_lag_order(df_covid, df_mental):
    df = concat([df_mental, df_covid], axis=1)
    model = VAR(df)
    for i in [1,2,3,4,5,6,7,8,9,10]:
        result = model.fit(i)
        print('Lag Order =', i)
        print('AIC : ', result.aic)
        print('BIC : ', result.bic)
        print('FPE : ', result.fpe)
        print('HQIC: ', result.hqic, '\n')
    return model, df

def fit_model(df, model, lag=8):
    model_fitted = model.fit(8)
    print(model_fitted.summary())
    lag_order = model_fitted.k_ar
    print(lag_order)  #> 4
    model_fitted.forecast(df.values[-lag_order:], 15)
    model_fitted.plot_forecast(20)
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

def previsao_ansiedade(df_covid, df_mental, inicio):
    coefs = [-0.497103, -2.093935, -0.433451, -1.403782, -0.425816, -0.288431, -0.301580, 1.255962, 0.852093]
    lags = [1, 1, 2, 2, 3, 4, 5, 6, 7]
    curvas = ['mental', 'covid', 'mental', 'covid', 'mental', 'mental', 'mental', 'covid', 'covid']
    df_ansiedade = DataFrame(columns=df_mental.columns)
    for i, data in enumerate(df_mental.index[inicio:]):
        parcial_sum = 0
        for coef, lag, curva in zip(coefs, lags, curvas):
            parcial_sum += coef * eval('df_'+curva).iloc[i-lag][0]
        df_ansiedade.loc[data] = parcial_sum
    plt.plot(df_mental)
    plt.plot(df_ansiedade)
    plt.show()

df_covid, df_mental = testes_google_trends()
model, df = find_lag_order(df_covid, df_mental)
fit_model(df, model)
#previsao_ansiedade(df_covid, df_mental, 10)

#correlacoes = []
#for doenca in ['ansiedade', 'depressao', 'ambos']:
#    for sigla, estado in estados:
#        pearson, spearman = computa_curvas(sigla, estado, doenca, plot=False, estadual=True)
#        correlacoes.append((estado, doenca, pearson, spearman))
#
#correlacoes_ordenadas = sorted(correlacoes, key=lambda item : item[3], reverse=True)
#for estado, doenca, pearson, spearman in correlacoes_ordenadas:
#    print(estado, doenca, pearson, spearman, end='\n\n')
