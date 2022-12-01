import numpy
import pickle
import os

from loaders.krx_loader import Dataset_krx
from loaders.fi2010_loader import Dataset_fi2010
from loggers import logger


def __get_dataset__(model_id, dataset_type, normalization, lighten, T, k, stock, test_days):
    if dataset_type == 'fi2010':
        auction = False
    elif dataset_type == 'krx':
        lighten = True

    if dataset_type == 'fi2010':
        dataset_test = Dataset_fi2010(auction, 'DecPre', stock, test_days, T, k, lighten)
    elif dataset_type == 'krx':
        dataset_test = Dataset_krx('DecPre', stock, test_days, T, k)
    else:
        print("Error: wrong dataset type")

    return dataset_test


def __get_midprice__(model_id, dataset_type, normalization, lighten, T, k, stock, test_days):
    dataset_test = __get_dataset__(model_id, dataset_type, normalization, lighten, T, k, stock, test_days)
    midprice = []
    for i in range(dataset_test.__len__()):
        x, _ = dataset_test.__getitem__(i)
        ask_price = x[:, -1, 0]
        bid_price = x[:, -1, 2]
        temp_midprice = ask_price + bid_price
        midprice.append(float(temp_midprice))
    return midprice


def __get_prediction__(model_id):
    with open(os.path.join(logger.find_save_path(model_id), 'prediction.pkl'), 'rb') as f:
        all_targets, all_predictions = pickle.load(f)
    return all_targets, all_predictions


def backtest(model_id, dataset_type, normalization, lighten, T, k, stock, test_days):
    midprice = __get_midprice__(model_id, dataset_type, normalization, lighten, T, k, stock, test_days)
    all_targets, all_predictions = __get_prediction__(model_id)
    print(len(midprice))
    print(len(all_targets))
    print(len(all_predictions))