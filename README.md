# Air Quality in London
---------------------------------<i>Work in Progress</i>-----------------------------

This repo contains a study of Nitrogen Oxide (NO2) emissions in London.

Emissions data was downloaded from the  <a href="https://www.londonair.org.uk/Londonair/API/" target="_blank">Environmental Research Group API</a>, which monitors polluton levels at over 200 sites around London. Five sites with sufficient historical data were randomly selected for the study.

Traffic is the main source of pollution in London and traffic counts are available <a href="https://roadtraffic.dft.gov.uk/downloads" target="_blank">here</a>. However, data is not at the required level of granularity (hourly) to be used in this study.

Weather can also affect pollution levels, especially wind and rain, which remove it from the air. Historical UK weather data is available from the <a href="https://roadtraffic.dft.gov.uk/downloads" target="_blank">CEDA</a> archive. One weather station (Heathrow) was chosen to represent London.

Emission levels over time are shown below, along with the main weather patterns. Unfortunately, some emissions data is missing for unknown reasons.

![Emissions over time](/plots/data_raw.png)

Selected site locations are shown below, with Islington as the target site.

![Site locations](/plots/site-locations.png)

The goal of the analysis was to predict Nitrogen Oxide (NO2) emissions for Islington using emissions other four sites (Enfield, Newham, Greenwich and Redbridge) along with weather data. Being able to predict pollution for a different location could reduce the cost of further scanning sites.

### Missing data

As shown above, there are gaps in the pollution time series. The amount of missing data is shown in the below table. Dropping all the missing data would result in the loss of ~40% of data. It was decided to drop the missing time points from the target site (Islington) only, so as to compare results more easily. For simplicity, the other missing values were imputed using the means of the time series.

| Item | % Missing |
| :----|----------:|
|MeasurementDateGMT                                             | 0.00|
|Enfield - Bowes Primary School: Nitrogen Dioxide (ug/m3)       |21.41|
|Newham - Cam Road: Nitrogen Dioxide (ug/m3)                     |7.49|
|Greenwich - Plumstead High Street: Nitrogen Dioxide (ug/m3)     |6.13|
|Redbridge - Gardner Close: Nitrogen Dioxide (ug/m3)            |13.55|
|Islington - Holloway Road: Nitrogen Dioxide (ug/m3)            | 6.03|
|wind_direction                                                 | 0.71|
|wind_speed                                                     | 0.71|
|msl_pressure                                                   | 0.12|
|air_temperature                                                | 0.08|

### Train-test split

It was decided to use the years [2010-2016] as training data, [2017-2018] as validation data,
and [2019] as test data. Further time-series cross-validation was done on the training data.

### Baseline model

A random forest model was used as a baseline prediction. This treated the data points as independent and did not take the time series into account, apart from the cross-validation split. A randomised search was performed to set appropriate hyperparameters. This resulted in a forest of 146 trees using a max of 10 features and a max depth of 28. Metrics and graphs are shown below.

| Metric | Training | Validation |
| :--- | ---:|---:|
| Mean Absolute Error |3.40 |8.47 |
| Mean Squared Error |21.32 |121.91 |
| Root Mean Squared Error |4.62 |11.04 |
| Residual Mean |-0.06 |-2.91 |
| Residual Median |-0.47 |-3.70 |
| Residual Standard Deviation |4.62 |10.65 |

