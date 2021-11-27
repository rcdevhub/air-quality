# Air Quality in London
---------------------------------<i>Work in Progress</i>-----------------------------

This repo contains a study of Nitrogen Oxide (NO2) emissions in London. Air pollution in London has been a topic of environmental and political concern recently, and has lead to the introduction of the Ultra Low Emission Zone (ULEZ).

Emissions data was downloaded from the  <a href="https://www.londonair.org.uk/Londonair/API/" target="_blank">Environmental Research Group API</a>, which monitors polluton levels at over 200 sites around London. Five sites with sufficient historical data were randomly selected for the study.

Traffic is the main source of pollution in London and traffic counts are available <a href="https://roadtraffic.dft.gov.uk/downloads" target="_blank">here</a>. However, data is not at the required level of granularity (hourly) to be used in this study.

Weather can also affect pollution levels, especially wind and rain, which remove it from the air. Historical UK weather data is available from the <a href="https://roadtraffic.dft.gov.uk/downloads" target="_blank">CEDA</a> archive. One weather station (Heathrow) was chosen to represent London.

Emission levels over ten years are shown below, along with the main weather patterns. Unfortunately, some emissions data is missing for unknown reasons.

![Emissions over time](/plots/data_raw.png)

Selected site locations are shown below, with Islington as the target site.

![Site locations](/plots/site-locations.png)

The goal of the analysis was to predict Nitrogen Oxide (NO2) emissions for Islington using emissions other four sites (Enfield, Newham, Greenwich and Redbridge) along with weather data. Being able to predict pollution for a different location could reduce the cost of further scanning sites.

### Missing data

As shown above, there are gaps in the pollution time series. The amount of missing data is shown in the below table. The reason for the missing data is unknown and was assumed to be random. Dropping all the missing data would result in the loss of ~40% of data. It was decided to drop the missing time points from the target site (Islington) only, so as to compare results more easily. For simplicity, the other missing values were imputed using the means of the time series.

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

### Trends

Seasonal trends are somwhat unclear from the data. Autocorrelation and partial autocorrelation plots on the target time series are shown below and indicate strong time-based correlations, with daily peaks at 24 and 48 hours.

<img src="plots/isl_acf.png" width="420" height="280" /><img src="plots/isl_pacf.png" width="420" height="280" />

### Train-test split

It was decided to use the years [2010-2016] as training data, [2017-2018] as validation data,
and [2019] as test data. Further time-series cross-validation was done on the training data.

### Baseline model

A random forest model was used as a baseline prediction. This treated the data points as independent and did not take the time series into account, apart from the cross-validation split. A randomised search was performed to set appropriate hyperparameters. This resulted in a forest of 101 trees using a max of 4 features and a max depth of 34. Metrics and graphs are shown below, with extracts from the time series. Note the daily morning and evening rush hours. The model does not capture these local spikes particularly accurately, and the residuals are biased for the validation data from [2017-2018], possibly indicating a long-term traffic trend over years that is not captured.

| Metric | Training | Validation |
| :--- | ---:|---:|
| Mean Absolute Error |3.44 |8.63 |
| Mean Squared Error |22.05 |123.76 |
| Root Mean Squared Error |4.70 |11.12 |
| Residual Mean |-0.05 |-3.54 |
| Residual Median |-0.47 |-4.14 |
| Residual Standard Deviation |4.70 |10.55 |

<img src="plots/pred-act-base-training-.png" width="420" height="280" /><img src="plots/pred-act-base-validation-.png" width="420" height="280" />
<img src="plots/resid-base-training-.png" width="420" height="280" /><img src="plots/resid-base-validation-.png" width="420" height="280" />
<img src="plots/pred-time-base-validation-2017-03-06-2017-03-09.png" width="480" height="240" /><img src="plots/pred-time-base-validation-2017-05-01-2017-05-15.png" width="480" height="240" />

### Time series model - Regression

A set of models were fitted using sliding windows of 80 hours on the input time series. This was to capture the daily patterns of traffic. An important dynamic is the morning and evening rush hours on weekdays. Firstly a ridge regression model was fitted. This did not perform as well as the baseline model and again had biased residuals in the validation data.

| Metric | Training | Validation |
| :--- | ---:|---:|
| Mean Absolute Error |11.75 |10.18 |
| Mean Squared Error |239.21 |165.03 |
| Root Mean Squared Error |15.47 |12.85 |
| Residual Mean |-0.00 |-4.54 |
| Residual Median |-1.54 |-5.38 |
| Residual Standard Deviation |15.47 |12.02 |

<img src="plots/pred-act-rdg-training-.png" width="420" height="280" /><img src="plots/pred-act-rdg-validation-.png" width="420" height="280" />
<img src="plots/resid-rdg-training-.png" width="420" height="280" /><img src="plots/resid-rdg-validation-.png" width="420" height="280" />
<img src="plots/pred-time-rdg-validation-2017-03-06-2017-03-09.png" width="480" height="240" /><img src="plots/pred-time-rdg-validation-2017-05-01-2017-05-15.png" width="480" height="240" />

### Time series model - Random forest

A random forest model was fitted, which should be able to better capture non-linear trends. A randomised search was performed, resulting in a model with 61 trees using a max of 77 features and a max depth of 53. The total number of features was 1280 (80 time steps for each of 8 features, doubled for imputation flags). The metrics for the random forest are slightly better than the regression model, but not as good as the baseline. In addition, the hourly accuracy does not track the peaks as closely as the baseline does, indicating that the point-in-time information that the baseline model uses is not being improved by the time series history.

| Metric | Training | Validation |
| :--- | ---:|---:|
| Mean Absolute Error |8.44 |10.39|
| Mean Squared Error |129.06 |162.82|
| Root Mean Squared Error |11.36 |12.76|
| Residual Mean |-0.07 |-5.32 |
| Residual Median |-1.48 |-6.55 |
| Residual Standard Deviation |11.36 |11.6|

<img src="plots/pred-act-rf-ts-training-.png" width="420" height="280" /><img src="plots/pred-act-rf-ts-validation-.png" width="420" height="280" />
<img src="plots/resid-rf-ts-training-.png" width="420" height="280" /><img src="plots/resid-rf-ts-validation-.png" width="420" height="280" />
<img src="plots/pred-time-rf-ts-validation-2017-03-06-2017-03-09.png" width="480" height="240" /><img src="plots/pred-time-rf-ts-validation-2017-05-01-2017-05-15.png" width="480" height="240" />

### Feature Engineering

Several features were added in an attempt to improve model performance. Feature importance metrics showed that wind direction was more important than wind speed. This could be because Islington is located in North London and so north winds would be blowing clean air down, rather than polluted air carried by the prevailing south-west wind across London. A multiplicative interaction between wind speed and wind direction was introduced. Also, the time series window of 80 hours was not long enough to capture long-term trends. Chronological features were added, that is, hour of day, day of week, week of year and year. These were added as numerical variables.

The baseline random forest model (non-time series) was the best performing model so far, and so this was taken forward for use with the new features. A new randomised search was performed, resulting in a model with 145 trees, with max features of 5 and a max depth of 39. Metrics and graphs are shown below. The new features appeared to slightly improve performance on the training set, but worsened it on the validation set, indicating overfitting.

| Metric | Training | Validation |
| :--- | ---:|---:|
| Mean Absolute Error |2.86 |9.80|
| Mean Squared Error |15.76 |143.75|
| Root Mean Squared Error |3.97 |11.99|
| Residual Mean |-0.05 |-7.32 |
| Residual Median |-0.42 |-7.72 |
| Residual Standard Deviation |3.97 |9.49|

<img src="plots/pred-act-reg-feat-training-.png" width="420" height="280" /><img src="plots/pred-act-reg-feat-validation-.png" width="420" height="280" />
<img src="plots/resid-reg-feat-training-.png" width="420" height="280" /><img src="plots/resid-reg-feat-validation-.png" width="420" height="280" />
<img src="plots/pred-time-reg-feat-validation-2017-03-06-2017-03-09.png" width="480" height="240" /><img src="plots/pred-time-reg-feat-validation-2017-05-01-2017-05-15.png" width="480" height="240" />
