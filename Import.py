# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 22:53:53 2021

@author: rcpc4
"""

#--------------------------Imports----------------------------------

import os
import sys

os.chdir('C://Code/Projects/air-quality')

import io
import json
import requests
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error)

import plotly.express as px
from plotly.offline import plot

from statsmodels.graphics.tsaplots import (plot_acf, plot_pacf)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pytorch_forecasting.data import TimeSeriesDataSet

#--------------------------API functions----------------------------------

def get_sites():
    '''Get site list with details.'''
    base_url = 'http://api.erg.ic.ac.uk/AirQuality'
    target_url = '/Information/MonitoringSites/GroupName=London/Json'
    url = base_url + target_url
    
    response = requests.request("GET",url)
    sites = json.loads(response.text)
    sites_df = pd.DataFrame.from_dict(sites['Sites']['Site'])
    sites_df['@DateOpened'] = pd.to_datetime(sites_df['@DateOpened'])
    sites_df['@DateClosed'] = pd.to_datetime(sites_df['@DateClosed'])
    sites_df['@Latitude'] = pd.to_numeric(sites_df['@Latitude'])
    sites_df['@Longitude'] = pd.to_numeric(sites_df['@Longitude'])
    
    return sites_df

def get_obs_data(sitecode,speciescode,startdate,enddate):
    '''Get raw observational data.'''
    base_url = 'http://api.erg.ic.ac.uk/AirQuality'
    url = str(base_url+'/Data/SiteSpecies/SiteCode='+sitecode+
              '/SpeciesCode='+speciescode+
              '/StartDate='+startdate+'/EndDate='+enddate+'/csv')
    response = requests.request("GET",url)
    data = pd.read_csv(io.StringIO(response.text))
    
    return data

def get_traf_data(sitecode,startdate,enddate):
    '''Get traffic data.'''
    # Note: unused as it appears no sites have traffic data
    base_url = 'http://api.erg.ic.ac.uk/AirQuality'
    url = str(base_url+'/Data/Traffic/Site/SiteCode='+sitecode+
              '/StartDate='+startdate+'/EndDate='+enddate+'/Json')
    response = requests.request("GET",url)
    traffic = json.loads(response.text)
    
    return traffic

#--------------------------Data import functions-------------------------

def get_weather_data():
    '''Get weather data from saved files.'''
    # Source: https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-202107/
    weath_years = [*range(2010,2021,1)]
    weath_path_a = 'data/weather/midas-open_uk-hourly-weather-obs_dv-202107_greater-london_00708_heathrow_qcv-1_'
    weath_cols=['ob_time','wind_direction','wind_speed','air_temperature','msl_pressure']
    data = []
    for year in weath_years:
        yeardata = pd.read_csv(weath_path_a+str(year)+'.csv',
                               skiprows=280,
                               usecols=weath_cols)
        data.append(yeardata)
        
    all_data = pd.concat(data,axis=0)
    all_data = all_data[all_data['ob_time'] != 'end data'].reset_index(drop=True)
    all_data['ob_time'] = pd.to_datetime(all_data['ob_time'])
    
    return all_data

#--------------------------Modelling functions-------------------------

def compute_reg_metrics(Y_train,pred_train):
    '''Compute regession metrics for predictions'''
    metrics = {}
    metrics['mse'] = mean_squared_error(Y_train,pred_train)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(Y_train, pred_train)
    metrics['mape'] = mean_absolute_percentage_error(Y_train, pred_train)
    
    return metrics

#--------------------------Plotting functions-------------------------

def plot_feat_imp(model,names):
    '''Plot feature importance from rf model.'''
    feat_imp = pd.Series(model.feature_importances_,
                         index=names)
    plt.figure()
    feat_imp.sort_values(ascending=False).plot.bar()
    
    return feat_imp

def plot_resid(residuals):
    '''Plot regression residuals.'''
    plt.figure()
    sns.histplot(residuals)
    plt.axvline(x=0,linewidth=1,color='black')
    return None

def plot_pred_vs_act(Y_true,Y_pred,bins,x_low=None,x_high=None,y_low=None,y_high=None):
    '''Plot regression predicted vs actual density.'''
    fig, ax = plt.subplots(1,1)
    plt.hist2d(x=Y_true,y=Y_pred,bins=bins,cmap=plt.cm.jet)
    plt.plot(Y_true,Y_true,linestyle='-',linewidth=0.03,color='white')
    plt.colorbar()
    if (x_low is not None) & (x_high is not None):
        ax.set_xlim([x_low,x_high])
    if (y_low is not None) & (y_high is not None):
        ax.set_ylim([y_low,y_high])
    plt.title('Predicted vs Actual density')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    return None

def plot_pred_time(dates,Y_true,Y_pred,date_low=None,date_high=None):
    '''Plot predicted and actual time series between two dates.'''
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(dates,Y_true,linewidth=0.5,label='Actual')
    ax.plot(dates,Y_pred,linewidth=0.5,label='Predicted')
    if (date_low is not None) & (date_high is not None):
        ax.set_xlim([date_low,date_high])
    plt.legend()
    return None

def plot_resid_box(residuals,variable):
    '''Plot residuals boxplot by variable.'''
    plt.figure()
    sns.boxplot(variable,residuals,color='lightcyan')
    plt.axhline(y=0,color='black',linewidth=1)
    plt.ylabel('Residual')
    return None
    
#--------------------------Get data----------------------------------
    
# Get sitelist
sites = get_sites()
sites.to_pickle('data/pollution/sites.pkl')

# Check which sites have NO2 data
data = {}
for i in sites['@SiteCode']:
    data[i] = get_obs_data(i,'NO2','01Jan2020','02Jan2020')

records = pd.DataFrame(index=sites['@SiteCode'])
records['has_data'] = False
for i in data:
    records.loc[i,'has_data'] = not np.isnan(data[i].iloc[0,1])

# Select possible sites for analysis
# Select all type=='Roadside' as type strongly affects pollution
sites_with_data = sites[np.array(records['has_data'])]
sites_candidate = sites[(np.array(records['has_data']))
                        &(sites['@SiteType']=='Roadside')
                        &(sites['@DateOpened']<datetime(2010,1,1))
                        &(sites['@DateClosed'].isnull())]

# Select specific sites at random
selected_sites = sites_candidate.sample(n=5,random_state=100)
selected_sitecodes = selected_sites['@SiteCode']

# Get full data
full_data = {}
for i in selected_sitecodes:
    full_data[i] = get_obs_data(i,'NO2','01Jan2010','31Dec2019')

# Check all dates are all the same as first set of dates
for i in full_data:
    assert np.array_equal(full_data[i]['MeasurementDateGMT'],
                      full_data[selected_sitecodes.iloc[0]]['MeasurementDateGMT']), "Unequal date range"

# Join into one frame
data_nit = full_data[selected_sitecodes.iloc[0]]
for i in range(1,selected_sitecodes.shape[0]):
    data_nit = pd.merge(data_nit,full_data[selected_sitecodes.iloc[i]],
                        on='MeasurementDateGMT')
data_nit['MeasurementDateGMT'] = pd.to_datetime(data_nit['MeasurementDateGMT'])
    
# Weather
weath1 = get_weather_data()

# Join weather to pollution
data = pd.merge(data_nit,weath1,
                left_on='MeasurementDateGMT',
                right_on='ob_time',
                how='left')
data = data.drop('ob_time',axis=1)

#--------------------------Explore data----------------------------------

captions = data_nit.columns[1:]

# Plot selected sites
fig, axes = plt.subplots(5,1)
for i in range(captions.shape[0]):
    axes[i].plot(data_nit[captions[i]])
    axes[i].set_title(captions[i])
plt.tight_layout()

# Plot weather
fig, axes = plt.subplots(3,1)
axes[0].plot(weath1['wind_speed'])
axes[1].plot(weath1['msl_pressure'])
axes[2].plot(weath1['air_temperature'])
axes[0].set_title('wind_speed')
axes[1].set_title('msl_pressure')
axes[2].set_title('air_temperature')
for i in range(3):
    axes[i].set_xticks(ticks=[])
plt.tight_layout()

# Plot joint
fig, axes = plt.subplots(8,1,figsize=(20,10))
for i in range(captions.shape[0]):
    axes[i].plot(data[captions[i]])
    axes[i].set_title(captions[i])
axes[5].plot(data['wind_speed'])
axes[6].plot(data['msl_pressure'])
axes[7].plot(data['air_temperature'])
axes[5].set_title('wind_speed')
axes[6].set_title('msl_pressure')
axes[7].set_title('air_temperature')
for i in range(8):
    axes[i].set_xticks(ticks=[])
plt.xticks(ticks=range(data.shape[0]),
           labels=data['MeasurementDateGMT'].dt.strftime('%Y-%b'),
           fontsize=10)
plt.locator_params(nbins=11) # Automatically reduces xticks
plt.tight_layout()
plt.savefig('plots/data_raw.png',format='png',dpi=300)

# Plot detail
date_min = datetime(2010,7,1)
date_max = datetime(2010,7,8)
date_min_ind = np.argwhere((data['MeasurementDateGMT']==date_min).to_numpy())
date_max_ind = np.argwhere((data['MeasurementDateGMT']==date_max).to_numpy())
target_date_mask = ((data['MeasurementDateGMT']>=date_min)
                    &(data['MeasurementDateGMT']<date_max)).to_numpy()

fig, axes = plt.subplots(8,1,figsize=(20,10))
for i in range(captions.shape[0]):
    axes[i].plot(data[captions[i]])
    axes[i].set_title(captions[i])
axes[5].plot(data['wind_speed'])
axes[6].plot(data['msl_pressure'])
axes[7].plot(data['air_temperature'])
axes[5].set_title('wind_speed')
axes[6].set_title('msl_pressure')
axes[7].set_title('air_temperature')
for i in range(8):
    axes[i].set_xticks(ticks=[])
    axes[i].set_xlim(date_min_ind,date_max_ind)
plt.xticks(ticks=np.array([*range(0,data.shape[0])])[target_date_mask],
           labels=data.loc[target_date_mask,
                           'MeasurementDateGMT'].dt.strftime('%Y-%b-%d'),
           fontsize=10)
plt.locator_params(nbins=10) # Automatically reduces xticks
plt.tight_layout()
plt.savefig('plots/data_detail.png',format='png',dpi=300)

# Plot selected locations on map
fig = px.scatter_mapbox(selected_sites,
                        lat='@Latitude',
                        lon='@Longitude',
                        hover_name='@SiteName',
                        size=8*np.ones(selected_sites.shape[0]),
                        zoom=10,
                        width=800,
                        height=800,
                        title='Site Locations',
                        color=(selected_sites['@SiteName']=='Islington - Holloway Road'))
fig.update_layout(mapbox_style='open-street-map')
plot(fig)

# Count missing values
na_locs = data.isna()
na_count = na_locs.sum()
na_pct = na_count/data.shape[0]
# Dropping every na row results in loss of ~40% of data
data_nona = data.dropna()
na_pct_total = 1-data_nona.shape[0]/data.shape[0]

#--------------------------Preprocess data----------------------------------

# Arbitrarily select Islington - Holloway Road as target site
# Remove records where target variable missing (~6%)
# For clarity in model comparison. Other missing variables imputed (later).
data_cut = data.loc[data['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'].isna()==False]

# Train-test split
# Choose [2010-2016] as training data, [2017-2018] as validation data,
# [2019] as test data
data_train = data_cut.loc[data['MeasurementDateGMT']<datetime(2018,1,1)]
data_valid = data_cut.loc[(data['MeasurementDateGMT']>=datetime(2018,1,1))
                          & (data['MeasurementDateGMT']<datetime(2019,1,1))]
data_test = data_cut.loc[data['MeasurementDateGMT']>=datetime(2019,1,1)]

X_train = data_train.drop(['MeasurementDateGMT',
                           'Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'],
                          axis=1).values
Y_train = data_train['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'].values

var_names = data_train.drop(['MeasurementDateGMT',
                             'Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'],
                            axis=1).columns

X_valid = data_valid.drop(['MeasurementDateGMT',
                           'Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'],
                          axis=1).values
Y_valid = data_valid['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'].values

# Test TBC

#--------------------------Baseline models----------------------------------

# Models
# Random forest, ignoring time component
regressors = {'rf': RandomForestRegressor()}

hyperparameters = {'rf':{'regressor__n_estimators':[100],
                         'regressor__max_depth':[*range(1,50,1)],
                         'regressor__max_features':[*range(1,10,1)],
                         'regressor__min_samples_split':[2],
                         'regressor__min_samples_leaf':[1],
                         'regressor__max_samples':[None]
                             }}

# Replace missing values with mean
imputer = SimpleImputer(missing_values=np.nan,
                        strategy='mean',
                        add_indicator=True)
# Normalise
scaler = StandardScaler()
# Time-based cv
splitter = TimeSeriesSplit(n_splits=5)

pipe = Pipeline([('imputer',imputer),
                 ('scaler',scaler),
                 ('regressor',regressors['rf'])])

random_search_iter = 20
score_metric = 'neg_mean_squared_error'

reg_cv = RandomizedSearchCV(estimator=pipe,
                            param_distributions=hyperparameters['rf'],
                            n_iter=random_search_iter,
                            scoring=score_metric,
                            cv=splitter,
                            verbose=3)

reg_cv.fit(X_train,Y_train)
reg_cv_results = reg_cv.cv_results_
reg_cv_best = reg_cv.best_estimator_
# Refit on all training data
reg_cv_best.fit(X_train,Y_train)
pred_base_train = reg_cv_best.predict(X_train)
pred_base_valid = reg_cv_best.predict(X_valid)
resid_base_train = Y_train - pred_base_train
resid_base_valid = Y_valid - pred_base_valid
# Compute metrics
pred_base_metrics_train = compute_reg_metrics(Y_train,pred_base_train)
pred_base_metrics_valid = compute_reg_metrics(Y_valid,pred_base_valid)

var_names_imputed = [*var_names, *[i+'_imputed' for i in var_names]]
# Plot diagnostics
plot_feat_imp(reg_cv_best.named_steps['regressor'],var_names_imputed)
plot_resid(resid_base_train)
plot_pred_vs_act(Y_train, pred_base_train, 200, x_low=0, x_high=100, y_low=0, y_high=100)
# Plot predictions
plot_pred_time(data_train['MeasurementDateGMT'],Y_train,pred_base_train,
               date_low=datetime(2016,10,1),date_high=datetime(2016,10,31))
# Plot residuals
plot_resid_box(resid_base_train,pd.DatetimeIndex(data_train['MeasurementDateGMT']).year)

#--------------------------Time Series EDA---------------------------

# Plot ACF
# Note importance of recent readings and daily peaks at 24 and 48 hours
# Note the confidence interval is there but nearly invisible
plot_acf(data_train['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'])
# plt.ylim([-0.25,0.25])

# Plot PACF
# Shows the most recent two hours as critical, and 24, 48 hours as important
# Again note these are all outside the invisible confidence interval
# Subsequent peaks indicate that ARIMA model unlikely to be sufficient
plot_pacf(data_train['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'])
# plt.ylim([-0.1,0.1])

#--------------------------ARIMA------------------------------

# Prepare data for ARIMA model
tseries_train = data_train['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)']
tseries_train.index = data_train['MeasurementDateGMT']

# Fit AR(2) model (assumes no trend or seasonality)
ar2 = ARIMA(tseries_train,order=(2,0,0))
ar2_fit = ar2.fit()
print(ar2_fit.summary())
pred_ar2_train = ar2_fit.predict()
# The model basically predicts using the last value and a bit of info from
# the previous value. Basically persistence.
# May also be a small amount of skew from the discontinuities
# Residuals are less biased than the non-ts baseline above
pred_ar2_metrics_train = compute_reg_metrics(Y_train,pred_ar2_train.values)
resid_ar2_train = pd.DataFrame(ar2_fit.resid)
print(resid_ar2_train.describe())
resid_ar2_train.plot()
plot_resid(resid_ar2_train)
# Note the ACF graph of the residuals has lines way outside the confidence
# interval, indicating that model is not fully explaining behaviour
plot_acf(resid_ar2_train)
plt.ylim([-0.1,0.1])
plot_pred_vs_act(Y_train, pred_ar2_train.values, 200, x_low=0, x_high=100, y_low=0, y_high=100)
# Pick a low range to see daily lag
plot_pred_time(data_train['MeasurementDateGMT'],Y_train,pred_ar2_train.values,
               date_low=datetime(2015,6,7),date_high=datetime(2015,6,15))
plot_resid_box(np.squeeze(resid_ar2_train.values),pd.DatetimeIndex(data_train['MeasurementDateGMT']).year)

#--------------------------SARIMA------------------------------

# AR(2) with 24-hour (daily) seasonality
# Still largely a persistence model, slightly improved RMSE, residuals now biased though
sar2_2_24 = SARIMAX(tseries_train,order=(2,0,0),seasonal_order=(2,0,0,24))
sar2_2_24_fit = sar2_2_24.fit()
print(sar2_2_24_fit.summary())
pred_sar2_2_24_train = sar2_2_24_fit.predict()
pred_sar2_2_24_metrics_train = compute_reg_metrics(Y_train,pred_sar2_2_24_train.values)
resid_sar2_2_24_train = pd.DataFrame(sar2_2_24_fit.resid)
print(resid_sar2_2_24_train.describe())
resid_sar2_2_24_train.plot()
plot_resid(resid_sar2_2_24_train)
plot_acf(resid_sar2_2_24_train)
plt.ylim([-0.1,0.1])
plot_pred_vs_act(Y_train, pred_sar2_2_24_train.values, 200, x_low=0, x_high=100, y_low=0, y_high=100)
plot_pred_time(data_train['MeasurementDateGMT'],Y_train,pred_sar2_2_24_train.values,
               date_low=datetime(2012,12,20),date_high=datetime(2012,12,31))
plot_resid_box(np.squeeze(resid_sar2_2_24_train.values),pd.DatetimeIndex(data_train['MeasurementDateGMT']).year)

#--------------------------Time Series Data Prep------------------------------

ts2 = data_train[['MeasurementDateGMT','Islington - Holloway Road: Nitrogen Dioxide (ug/m3)']].copy()

ts2.reset_index(inplace=True)
ts2.rename(columns={'index':'time_idx'},inplace=True)
ts2['group'] = np.zeros(ts2.shape[0])

tsd = TimeSeriesDataSet(ts2,
                        time_idx='time_idx',
                        target='Islington - Holloway Road: Nitrogen Dioxide (ug/m3)',
                        group_ids=['group'],
                        allow_missing_timesteps=True
                        )



#--------------------------Feature engineering------------------------------

