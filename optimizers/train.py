import torch
import yaml
import sys
import os
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary

from loaders.fi2010_loader import Dataset_fi2010
from loaders.krx_preprocess import get_normalized_data_list
from loaders.krx_loader import Dataset_krx
from models.deeplob import Deeplob
from optimizers.batch_gd import batch_gd

def train(dataset_type: str, normalization: str, lighten: bool,
          T: int, k: int, stock: list, train_test_ratio = float):

    if dataset_type == 'fi2010':
        auction = False
        days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif dataset_type == 'krx':
        lighten = True
        day_length = len(get_normalized_data_list(stock[0], normalization))
        days = list(range(day_length))

    train_day_length = round(len(days) * train_test_ratio)
    train_days = days[:train_day_length]
    test_days = days[train_day_length:]

    if dataset_type == 'fi2010':
        dataset_train_val = Dataset_fi2010(auction, normalization, stock, train_days, T, k, lighten)
        dataset_test = Dataset_fi2010(auction, normalization, stock, test_days, T, k, lighten)
    elif dataset_type == 'krx':
        dataset_train_val = Dataset_krx(normalization, stock, train_days, T, k)
        dataset_test = Dataset_krx(normalization, stock, test_days, T, k)
    else:
        print("Error: wrong dataset type")

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
    root_path = sys.path[0]
    with open(os.path.join(root_path, 'optimizers', 'hyperparams.yml'), 'r') as stream:
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

    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = batch_gd(model = model,
                                                                            criterion = criterion, optimizer = optimizer,
                                                                            train_loader = train_loader, val_loader = val_loader,
                                                                            epochs=50, name = model.name)

    print(train_loss_hist)
    print(train_acc_hist)
    print(val_loss_hist)
    print(val_acc_hist)

    return