# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 08:17:40 2021

@author: rcpc4
"""

# Imports
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

# Sequence class
# Credit: https://www.crosstab.io/articles/time-series-pytorch-lstm
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        
    def __len__(self):
        return self.X.shape[0]
    
    def num_feat(self):
        return len(self.features)
    
    def __getitem__(self,i):
        # Get sequence with target of index i (not included in sequence)
        # Pre-pads short sequences with the first value
        if i >= self.sequence_length:
            i_start = i - self.sequence_length
            x = self.X[i_start:i,:]
        else:
            padding = self.X[0].repeat(self.sequence_length-i,1)
            x = self.X[0:i,:]
            x = torch.cat((padding,x),0)
            
        return x,self.y[i]

# RNN model
class RNN_model(nn.Module):
    def __init__(self,num_features,num_hidden_units,num_layers):
        super().__init__()
        self.num_features = num_features
        self.num_hidden_units = num_hidden_units
        self.num_layers = num_layers
        
        self.rnn_layers = nn.RNN(input_size=self.num_features,
                                 hidden_size=self.num_hidden_units,
                                 num_layers=self.num_layers,
                                 batch_first=True)
        
        self.linear_layer = nn.Linear(in_features=self.num_hidden_units,
                                      out_features=1)
        
    def forward(self,x):
        
        rnn_out,h_n = self.rnn_layers(x)
        linear_out = self.linear_layer(h_n[0])
        
        return linear_out

def train_model(dataloader,model,loss_function,optimizer):
    '''Train the neural network model.'''
    num_batches = len(dataloader)
    total_loss = 0
    
    for X,y in dataloader:
        output = model(X)
        loss = loss_function(output,y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss/num_batches
    print(f'Train loss: {avg_loss}')
    
def valid_model(dataloader,model,loss_function):
    
    num_batches = len(dataloader)
    total_loss = 0
    
    with torch.no_grad():
        for X,y in dataloader:
            output = model(X)
            total_loss += loss_function(output,y).item()
            
    avg_loss = total_loss/num_batches
    print(f'Validation loss: {avg_loss}')
    
def model_predict(dataloader,model):
    '''Predict from trained model.'''
    
    output = torch.tensor([])
    with torch.no_grad():
        for X,_ in dataloader:
            pred = model(X)
            output = torch.cat((output,pred),dim=0)
            
    return output
    
# Prepare data - impute and scale
imputer = SimpleImputer()
scaler = StandardScaler()

X_train2 = imputer.fit_transform(X_train)
X_train2 = scaler.fit_transform(X_train2)

X_valid2 = imputer.transform(X_valid)
X_valid2 = scaler.transform(X_valid2)

# Put in dataframe for PyTorch dataset
data_nn_train = pd.DataFrame(X_train2,
                             columns=data_train.drop(['MeasurementDateGMT',
                                                      'Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'],
                                                     axis=1).columns)
data_nn_train['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'] = data_train['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)']

data_nn_valid = pd.DataFrame(X_valid2,
                             columns=data_valid.drop(['MeasurementDateGMT',
                                                      'Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'],
                                                     axis=1).columns)
data_nn_valid['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'] = data_valid['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)']

# Set up datasets
sequence_length = 80

tsds_train = SequenceDataset(data_nn_train,
                       target='Islington - Holloway Road: Nitrogen Dioxide (ug/m3)',
                       features=data_nn_train.drop(['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'],
                                                   axis=1).columns,
                       sequence_length=sequence_length)

tsds_valid = SequenceDataset(data_nn_valid,
                       target='Islington - Holloway Road: Nitrogen Dioxide (ug/m3)',
                       features=data_nn_valid.drop(['Islington - Holloway Road: Nitrogen Dioxide (ug/m3)'],
                                                   axis=1).columns,
                       sequence_length=sequence_length)

# Convert to dataloader
torch.manual_seed(456)

train_loader = DataLoader(tsds_train,batch_size=36,shuffle=False)

X,y = next(iter(train_loader))

# Define model

num_hidden_units = 16
num_features = tsds_train.num_feat()
num_layers = 3
learning_rate = 1e-3

model = RNN_model(num_features,num_hidden_units,num_layers)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)




