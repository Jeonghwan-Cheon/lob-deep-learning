import torch
import yaml
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary

from datasets.fi2010_loader import Dataset_fi2010
from models.deeplob import Deeplob
from batch_gd import batch_gd

############################################################
# Dataset setting
############################################################
auction = False
normalization = 'DecPre'
stock_idx = [0, 1, 2, 3, 4]
T = 100
k = 4
train_days = [1, 2, 3, 4, 5, 6, 7]
test_days = [8, 9, 10]
############################################################

dataset_train_val = Dataset_fi2010(auction, normalization, stock_idx, train_days, T, k)
dataset_size = dataset_train_val.__len__()
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size
dataset_train, dataset_val = random_split(dataset_train_val, [train_size, val_size])
del dataset_train_val

dataset_test = Dataset_fi2010(auction, normalization, stock_idx, test_days, T, k)

print(f"Training Data Size : {dataset_train.__len__()}")
print(f"Validation Data Size : {dataset_val.__len__()}")
print(f"Testing Data Size : {dataset_test.__len__()}")

model = Deeplob()
model.to(model.device)

summary(model, (1, 1, 100, 40))

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