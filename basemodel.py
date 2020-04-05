# -*- coding: utf-8 -*-
"""
Covid - 19  Forecast for North Macedonia
base model
Created on Sun Apr  5 17:00:29 2020

@author: Daniela Janeva
"""
import pandas as pd
from fbprophet import Prophet 

#%%

df =  pd.read_csv('covid19-MKD.csv')

confirmed =df.groupby('Date').sum()['Confirmed'].reset_index()

confirmed.columns = ['ds','y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])

confirmed

m = Prophet(interval_width = 0.95)
m.fit(confirmed)
future=m.make_future_dataframe(periods = 5)

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

confirmed_forecast_plot=m.plot(forecast)
confirmed_forecast_plot=m.plot_components(forecast)



