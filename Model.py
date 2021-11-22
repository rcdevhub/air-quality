# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:41:42 2021

@author: rcpc4
"""

# Imports

from pytorch_forecasting.data import TimeSeriesDataset

# Using PyTorch class
ts2 = data_train.drop(['MeasurementDateGMT'],
                      axis=1).copy()

ts2.reset_index(inplace=True)
ts2.rename(columns={'index':'time_idx'},inplace=True)

# Fill NA timesteps
ts_vals = ts2.values
imputer = SimpleImputer(missing_values=np.nan,
                        strategy='mean',
                        add_indicator=False)
imputer.fit(ts_vals)

ts_vals_imp = imputer.transform(ts_vals)
ts2 = pd.DataFrame(ts_vals_imp,
                   columns=ts2.columns,
                   index=ts2.index)

ts2['time_idx'] = ts2['time_idx'].astype(int)
ts2['group'] = 'wind_speed'

max_encoder_length = 80
max_prediction_length = 1

tsd = TimeSeriesDataSet(ts2,
                        time_idx='time_idx',
                        target='Islington - Holloway Road: Nitrogen Dioxide (ug/m3)',
                        group_ids=['group'],
                        allow_missing_timesteps=True,
                        max_encoder_length=max_encoder_length,
                        max_prediction_length=max_prediction_length
                        )

dataloader = tsd.to_dataloader(batch_size=36,train=False)

X, y = next(iter(dataloader))