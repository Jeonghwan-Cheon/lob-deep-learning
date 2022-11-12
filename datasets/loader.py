import os
import sys
import pandas as pd

def get_fi2010(auction = False, normalization = 'Zscore', dataset = 'Train'):
    """
    Load the FI2010 dataset
    Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine
    Learning Methods. A Ntakaris, M Magris, J Kanniainen, M Gabbouj, A Iosifidis.
    arXiv:1705.03233 [cs.CE].  https://arxiv.org/abs/1705.03233
    Parameters
    ----------
    auction : {True, False}
    normalization : {'Zscore', 'MinMax', 'DecPre'}
    normalization : {'Train', 'Test'}
    """

    root_path = sys.path[1]
    dataset_path = 'fi2010'

    if auction:
        path1 = "Auction"
    else:
        path1 = "NoAuction"

    if normalization == 'Zscore':
        tmp_path_1 = '1.'
        tmp_path_2 = path1 + '_' + normalization
    elif normalization == 'MinMax':
        tmp_path_1 = '2.'
        tmp_path_2 = path1 + '_' + normalization
    elif normalization == 'DecPre':
        tmp_path_1 = '3.'
        tmp_path_2 = path1 + '_' + normalization

    path2 = tmp_path_1 + tmp_path_2

    if dataset == 'Train':
        path3 = tmp_path_2 + '_' + 'Training'
    elif dataset == 'Test':
        path3 = tmp_path_2 + '_' + 'Testing'

    dataset_path = os.path.join(root_path, dataset_path, path1, path2, path3)
    item = sorted(os.listdir(dataset_path))

    fi2010_dataset = pd.read_csv(os.path.join(dataset_path, item[0]), index_col=0)
    if len(item) > 1:
        for i in range(len(item)-1):
            fi2010_dataset = pd.concat([\
                fi2010_dataset, pd.read_csv(os.path.join(dataset_path, item[i+1]), index_col=0)])

    return fi2010_dataset