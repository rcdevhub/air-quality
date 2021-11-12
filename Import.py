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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#--------------------------API functions----------------------------------

def get_sites():
    '''Get site list with details.'''
    base_url = 'http://api.erg.ic.ac.uk/AirQuality'
    target_url = '/Information/MonitoringSites/GroupName=London/Json'
    url = base_url + target_url
    
    response = requests.request("GET",url)
    sites = json.loads(response.text)
    sites_df = pd.DataFrame.from_dict(sites['Sites']['Site'])
    
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

#--------------------------Data import functions----------------------------------

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

#--------------------------Get data----------------------------------
    
# Get sitelist
sites = get_sites()

# Check which sites have NO2 data
data = {}
for i in sites['@SiteCode']:
    data[i] = get_obs_data(i,'NO2','01Jan2020','02Jan2020')

records = pd.DataFrame(index=sites['@SiteCode'])
records['has_data'] = False
for i in data:
    records.loc[i,'has_data'] = not np.isnan(data[i].iloc[0,1])

# Select possible sites for analysis    
sites_with_data = sites[np.array(records['has_data'])]
sites_candidate = sites[(np.array(records['has_data']))
                        &(sites['@SiteType']=='Roadside')
                        &(pd.to_datetime(sites['@DateOpened'])<datetime(2010,1,1))
                        &(sites['@DateClosed']=='')]

# Select sites at random
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
    
captions = data_nit.columns[1:]

# Plot to see gaps in data
fig, axes = plt.subplots(5,1)
for i in range(captions.shape[0]):
    axes[i].plot(data_nit[captions[i]])
    axes[i].title(captions[i])
plt.xticks(ticks=[*range(data_nit.shape[0])],labels=data_nit['MeasurementDateGMT'].to_list())

# Weather
weath1 = get_weather_data()

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

# Join weather to pollution
data = pd.merge(data_nit,weath1,
                left_on='MeasurementDateGMT',
                right_on='ob_time',
                how='left')

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
plt.tight_layout()
plt.savefig('plots/data_raw.png',format='png',dpi=300)

# Plot detail
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
    axes[i].set_xlim([1000,1600])
plt.xticks(ticks=range(1000,1600),
           labels=data.loc[range(1000,1600),'MeasurementDateGMT'].dt.strftime('%Y-%b'),
           fontsize=10)
plt.locator_params(nbins=10) # Automatically reduces xticks
plt.tight_layout()
plt.savefig('plots/data_detail.png',format='png',dpi=300)

#--------------------------Testing----------------------------------

# Plot full data
for i in full_data:
    plt.figure()
    full_data[i].plot()

test = pd.to_datetime(sites['@DateOpened'])

np.sum(pd.to_datetime(full_data['EN5']['MeasurementDateGMT']))

test2 = test > datetime(2011,1,3)

base_url = 'http://api.erg.ic.ac.uk/AirQuality'

# Map
target_url = '/Hourly/Map/Json'
url = base_url + target_url

response = requests.request("GET", url)

maps = response.text

maps2 = json.loads(maps)

# PDF
target_url = '/Information/Documentation/pdf'
url = base_url + target_url

with open('info.pdf','wb') as f:
    f.write(response.content)
    
# Species
target_url = '/Information/MonitoringSiteSpecies/GroupName=London/Json'
url = base_url + target_url

response = requests.request("GET",url)
species = json.loads(response.text)

# Groups
target_url = '/Information/Groups/Json'
url = base_url + target_url

response = requests.request("GET",url)
groups = json.loads(response.text)

# Sites


# Raw observations json - site
target_url = '/Data/Site/SiteCode=BQ5/StartDate=01Jan2019/EndDate=31Jan2019/Json'
url = base_url + target_url

response = requests.request("GET",url)
obs = json.loads(response.text)

# Raw observations json - site, species
target_url = '/Data/SiteSpecies/SiteCode=NB1/SpeciesCode=NO2/StartDate=01Jan2019/EndDate=31Jan2019/json'
url = base_url + target_url

response = requests.request("GET",url)
obs2 = json.loads(response.text)

# Raw observations csv - site, species
target_url = '/Data/SiteSpecies/SiteCode=NB1/SpeciesCode=NO2/StartDate=01Jan2019/EndDate=31Jan2019/csv'
url = base_url + target_url

response = requests.request("GET",url)
with open('response.csv','wb') as f:
    f.write(response.content)

obs2 = pd.read_csv('response.csv')
obs3 = pd.read_csv(io.StringIO(response.text))



