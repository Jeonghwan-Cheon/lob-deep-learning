import numpy
import pickle
import os
import matplotlib.pyplot as plt

from loaders.krx_loader import Dataset_krx
from loaders.fi2010_loader import Dataset_fi2010
from loggers import logger

def __get_data__(model_id):
    with open(os.path.join(logger.find_save_path(model_id), 'prediction.pkl'), 'rb') as f:
        all_midprices, all_targets, all_predictions = pickle.load(f)
    return all_midprices, all_targets, all_predictions


def backtest(model_id):
    all_midprice, all_targets, all_predictions = __get_data__(model_id)
    print(all_predictions.shape)
    # plt.plot(list(range(len(all_midprice))), all_midprice)
    # plt.show()