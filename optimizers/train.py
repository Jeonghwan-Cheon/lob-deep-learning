import torch
import yaml
import sys
import os
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchinfo import summary

from loaders.fi2010_loader import Dataset_fi2010
from loaders.krx_preprocess import get_normalized_data_list
from loaders.krx_loader import Dataset_krx
from models.deeplob import Deeplob
from optimizers.batch_gd import batch_gd
from loggers import logger


def __get_dataset__(model_id, dataset_type, normalization, lighten, T, k, stock, train_test_ratio):
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
    elif dataset_type == 'krx':
        dataset_train_val = Dataset_krx(normalization, stock, train_days, T, k)
        dataset_test = Dataset_krx(normalization, stock, test_days, T, k)
    else:
        print("Error: wrong dataset type")

    dataset_size = dataset_train_val.__len__()
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size
    dataset_train, dataset_val = random_split(dataset_train_val, [train_size, val_size])
    dataset_val = dataset_test
    del dataset_train_val

    print(f"Training Data Size : {dataset_train.__len__()}")
    print(f"Validation Data Size : {dataset_val.__len__()}")

    dataset_info = {
        'dataset_type': dataset_type,
        'normalization': normalization,
        'lighten': lighten,
        'T': T,
        'k': k,
        'stock': stock,
        'train_days': train_days,
        'test_days': test_days
    }
    logger.logger(model_id, 'dataset_info', dataset_info)

    return dataset_train, dataset_val

def __get_hyperparams__(name):
    root_path = sys.path[0]
    with open(os.path.join(root_path, 'optimizers', 'hyperparams.yaml'), 'r') as stream:
        hyperparams = yaml.safe_load(stream)
    return hyperparams[name]

def train(model_id, dataset_type, normalization, lighten, T, k, stock, train_test_ratio):
    # get train and validation set
    dataset_train, dataset_val = __get_dataset__(model_id, dataset_type, normalization, lighten, T, k, stock, train_test_ratio)

    model = Deeplob(lighten=lighten)
    model.to(model.device)

    if lighten:
        feature_size = 20
    else:
        feature_size = 40

    summary(model, (1, 1, 100, feature_size))

    # Hyperparameter setting
    hyperparams = __get_hyperparams__(model.name)

    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    epoch = hyperparams['epoch']
    num_workers = hyperparams['num_workers']

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_gd(model_id = model_id, model = model, criterion = criterion, optimizer = optimizer,
             train_loader = train_loader, val_loader = val_loader, epochs=epoch, name = model.name)
    return

def continual_learning(id, model, days):
    pass
    return