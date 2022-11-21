import torch
import yaml
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary

from loaders.fi2010_loader import Dataset_fi2010
from loaders.krx_loader import Dataset_krx
from models.deeplob import Deeplob
from batch_gd import batch_gd

############################################################
# Dataset setting
############################################################
auction = False
normalization = 'Zscore'
T = 100
k = 4
lighten = False
dataset_type = 'krx'
stock = ['KS200', 'KQ150']
train_days = [1, 2, 3]
test_days = [4]
############################################################

if dataset_type == 'fi2010':
    dataset_train_val = Dataset_fi2010(auction, normalization, stock, train_days, T, k, lighten)
    dataset_test = Dataset_fi2010(auction, normalization, stock, test_days, T, k, lighten)
elif dataset_type == 'krx':
    dataset_train_val = Dataset_krx(normalization, stock, train_days, T, k)
    dataset_test = Dataset_krx(normalization, stock, test_days, T, k)

dataset_size = dataset_train_val.__len__()
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size
dataset_train, dataset_val = random_split(dataset_train_val, [train_size, val_size])
del dataset_train_val

print(f"Training Data Size : {dataset_train.__len__()}")
print(f"Validation Data Size : {dataset_val.__len__()}")
print(f"Testing Data Size : {dataset_test.__len__()}")

model = Deeplob(lighten=lighten)
model.to(model.device)

if lighten:
    feature_size = 20
else:
    feature_size = 40

summary(model, (1, 1, 100, feature_size))

############################################################
# Hyperparameter setting
############################################################
with open('hyperparams.yml', 'r') as stream:
    hyperparams = yaml.safe_load(stream)

batch_size = hyperparams[model.name]['batch_size']
learning_rate = hyperparams[model.name]['learning_rate']
epoch = hyperparams[model.name]['epoch']
############################################################

train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses, val_losses = batch_gd(model = model,
                                    criterion = criterion, optimizer = optimizer,
                                    train_loader = train_loader, val_loader = val_loader,
                                    epochs=50, name = model.name)