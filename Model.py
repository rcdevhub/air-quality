# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:41:42 2021

@author: rcpc4
"""

# Imports

import torch
from torch import nn

from pytorch_forecasting.data import TimeSeriesDataset
from pytorch_forecasting.models import BaseModel

# Prepare dataset to use PyTorch TimeSeriesDataset
ts2 = data_train.drop(['MeasurementDateGMT'],
                      axis=1).copy()

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
ts2['group'] = 'group1'
ts2.reset_index(inplace=True)
ts2.rename({'index':'time_idx'},axis=1,inplace=True)

# Stack to long format (incorrect)
# ts3 = ts2.stack(level=-1).reset_index().rename({'level_0':'time_idx',
#                                                 'level_1':'variable',
#                                                 0:'value'},
#                                                axis=1)
# ts3['variable'] = ts3['variable'].astype('category')

max_encoder_length = 80
max_prediction_length = 1

tsd = TimeSeriesDataSet(ts2,
                        time_idx='time_idx',
                        target='Islington - Holloway Road: Nitrogen Dioxide (ug/m3)',
                        group_ids=['group'],
                        allow_missing_timesteps=True,
                        max_encoder_length=max_encoder_length,
                        max_prediction_length=max_prediction_length,
                        time_varying_unknown_reals = ['Enfield - Bowes Primary School: Nitrogen Dioxide (ug/m3)',
                                                     'Newham - Cam Road: Nitrogen Dioxide (ug/m3)',
                                                     'Greenwich - Plumstead High Street: Nitrogen Dioxide (ug/m3)',
                                                     'Redbridge - Gardner Close: Nitrogen Dioxide (ug/m3)',
                                                     'Islington - Holloway Road: Nitrogen Dioxide (ug/m3)', 'wind_direction',
                                                     'wind_speed', 'msl_pressure', 'air_temperature']
                        )



# Note: to see which scalers and normalisers were applied, run: tsd.get_parameters()
# To run without these, set:
# target_normalizer=None,
# scalers={'Enfield - Bowes Primary School: Nitrogen Dioxide (ug/m3)': None,
#          'Newham - Cam Road: Nitrogen Dioxide (ug/m3)': None,
#          'Greenwich - Plumstead High Street: Nitrogen Dioxide (ug/m3)': None,
#          'Redbridge - Gardner Close: Nitrogen Dioxide (ug/m3)': None,
#          'wind_direction': None,
#          'wind_speed': None,
#          'msl_pressure': None,
#          'air_temperature': None}

# The dataloader generates batch tensors with dimensions:
# (batch size x window_length x number of features)
# To stop it shuffling, set: train=False
                       
dataloader = tsd.to_dataloader(batch_size=36,train=False)

X, y = next(iter(dataloader))

# Define test module
class FullyConnectedModule(nn.Module):
    def __init__(self, input_size:int, output_size:int, hidden_size:int,
                 n_hidden_layers:int, **kwargs):
        super().__init__()
        # Input layer
        module_list = [nn.Linear(input_size,hidden_size), nn.ReLU()]
        # Hidden layers
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size,hidden_size), nn.ReLU()])
        # Output layer
        module_list.append(nn.Linear(hidden_size,output_size))
        
        self.sequential = nn.Sequential(*module_list)
        
    def forward(self, x:torch.Tensor):
        return self.sequential(x)

# Define test model
class FullyConnectedModel(BaseModel):
    def __init__(self, input_size: int, output_size:int, hidden_size:int,
                 n_hidden_layers:int, **kwargs):
        self.save_hyperparameters()
        super().__init__(**kwargs)
        self.network = FullyConnectedModule(input_size=self.hparams.input_size,
                                            output_size=self.hparams.output_size,
                                            hidden_size=self.hparams.hidden_size,
                                            n_hidden_layers=self.hparams.n_hidden_layers
                                            )
    
    def forward(self, x: Dict[str,torch.Tensor]):
        # x is a batch generated based on the TimeSeriesDataset
        network_input = x['encoder_cont'].squeeze(-1)
        prediction = self.network(network_input).unsqueeze(-1)
        # rescale predictions into target space
        prediction = self.transform_output(prediction,target_scale=x['target_scale'])
        
        return self.to_network_output(prediction=prediction)
    
    @classmethod
    def from_dataset(cls,dataset:TimeSeriesDataset,**kwargs):
        new_kwargs={'output_size':dataset.max_prediction_length,
                    'input_size':dataset.max_encoder_length
                    }
        new_kwargs.update(kwargs)
        
        assert dataset.max_prediction_length == dataset.min_prediction_length, 'Decoder only supports a fixed length.'
        assert dataset.max_encoder_length == dataset.min_encoder_length, 'Encoder only supports a fixed length.'
        assert (len(dataset.time_varying_known_categoricals)==0
                and len(dataset.time_varying_known_reals)==0
                and len(dataset.time_varying_unknown_categoricals)==0
                and len(dataset.static_categoricals)==0
                and len(dataset.static_reals)==0
                and len(dataset.time_varying_unknown_reals)==1
                and dataset.time_varying_unknown_reals[0]==dataset.target
                ), 'Only covariate should be the target in time_varying_unknown_reals'
        
        return super().from_dataset(dataset, **new_kwargs)
    
model = FullyConnectedModel.from_dataset(tsd, hidden_size=10, n_hidden_layers=2)
model.summarize("full")