# GENERICOS
import pickle
import pandas as pd
import numpy as np
import itertools

# MODELOS
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.stl import STLForecast

# METRICAS
from sklearn.metrics import mean_squared_error


# DATOS
df_base = pd.read_csv('data/processed/data_processed_1990_2023.csv', index_col='fecha')

df_base.drop('target', axis=1, inplace=True)
df_base.index = pd.to_datetime(df_base.index)
tmed = df_base['tmed']

steps = 60
train = tmed[:-steps]
test  = tmed[-steps:]

# STLForecast + ARIMA
model_1 = STLForecast(train, ARIMA, model_kwargs=dict(order=(3,0,5), trend="t"))
fit_1 = model_1.fit()

# GUARDAR EL MODELO
with open('model/stl_arima.model', "wb") as archivo_salida:
    pickle.dump(fit_1, archivo_salida)

# SARIMAX
exo = df_base.drop('tmed', axis=1)[:-steps]
model_2 = SARIMAX(train, exog=exo, order=(3, 0, 5), seasonal_order=(0, 0, 0, 7))
fit_2 = model_2.fit()

# GUARDAR EL MODELO
with open('model/sarimax.model', "wb") as archivo_salida:
    pickle.dump(fit_2, archivo_salida)
