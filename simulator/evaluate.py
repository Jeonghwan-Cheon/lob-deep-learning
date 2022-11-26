import torch
import numpy as np
import sys
import os
import yaml
import pickle
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

from loaders.fi2010_loader import Dataset_fi2010
from loaders.krx_preprocess import get_normalized_data_list
from loaders.krx_loader import Dataset_krx
from loggers import logger
from models.deeplob import Deeplob

def __get_dataset__(model_id, dataset_type, normalization, lighten, T, k, stock, test_days):
    if dataset_type == 'fi2010':
        auction = False
    elif dataset_type == 'krx':
        lighten = True

    if dataset_type == 'fi2010':
        dataset_test = Dataset_fi2010(auction, normalization, stock, test_days, T, k, lighten)
    elif dataset_type == 'krx':
        dataset_test = Dataset_krx(normalization, stock, test_days, T, k)
    else:
        print("Error: wrong dataset type")

    print(f"Testing Data Size : {dataset_test.__len__()}")
    return dataset_test

def __get_hyperparams__(name):
    root_path = sys.path[0]
    with open(os.path.join(root_path, 'optimizers', 'hyperparams.yaml'), 'r') as stream:
        hyperparams = yaml.safe_load(stream)
    return hyperparams[name]

def test(model_id, custom_test_days = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(logger.find_save_path(model_id), 'best_val_model.pt'), map_location=device)

    new_model = Deeplob(lighten=True)
    new_model.load_state_dict(model.state_dict())
    model = new_model
    model.to(device)

    # dataset_info = logger.read_log(model_id, 'dataset_info')
    # dataset_type = dataset_info['dataset_type']
    # normalization = dataset_info['normalization']
    # lighten = dataset_info['lighten']
    # T = dataset_info['T']
    # k = dataset_info['k']
    # stock = dataset_info['stock']

    dataset_type = 'krx'
    normalization = 'Zscore'
    lighten = True
    T = 100
    k = 100
    stock = ['KS200', 'KQ150']

    # if custom_test_days == None:
    #     test_days = dataset_info['test_days']
    # else:
    #     test_days = custom_test_days

    test_days = [5, 6]

    dataset_test = __get_dataset__(model_id, dataset_type, normalization, lighten, T, k, stock, test_days)

    hyperparams = __get_hyperparams__(model.name)
    batch_size = hyperparams['batch_size']
    num_workers = hyperparams['num_workers']

    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_targets = []
    all_predictions = []
    all_outputs = []

    count = 0
    for inputs, targets in tqdm(test_loader):
        # Move to GPU
        model.eval()
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs)

        # Get prediction
        # torch.max returns both max and argmax
        max_output, predictions = torch.max(outputs, 1)

        # update counts
        all_targets.append(targets.to(device).numpy())
        all_predictions.append(predictions.to(device).numpy())
        all_outputs.append(max_output.to(device).numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    test_acc = accuracy_score(all_targets, all_predictions)

    with open(os.path.join(logger.find_save_path(model_id), 'prediction.pkl'), 'wb') as f:
        pickle.dump([all_targets, all_predictions, all_outputs], f)

    print(f"Test acc: {test_acc:.4f}")
    print(classification_report(all_targets, all_predictions, digits=4))

    return