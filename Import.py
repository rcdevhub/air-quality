# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 22:53:53 2021

@author: rcpc4
"""

import os
import sys

os.chdir('C://Code/Projects/air-quality')

import json
import requests

import pandas as pd

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
target_url = '/Information/MonitoringSites/GroupName=London/Json'
url = base_url + target_url

response = requests.request("GET",url)
sites = json.loads(response.text)

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

# Raw observations json - site, species
target_url = '/Data/SiteSpecies/SiteCode=NB1/SpeciesCode=NO2/StartDate=01Jan2019/EndDate=31Jan2019/csv'
url = base_url + target_url

response = requests.request("GET",url)
with open('response.csv','wb') as f:
    f.write(response.content)

obs2 = pd.read_csv('response.csv')

# Traffic
target_url = '/Data/Traffic/Site/SiteCode=NB1/StartDate=01Jan2019/EndDate=31Jan2019/json'
url = base_url + target_url

response = requests.request("GET",url)
traf = json.loads(response.text)
