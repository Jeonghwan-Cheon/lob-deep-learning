import os
import sys
import numpy as np
import torch

from krx_preprocess import get_normalized_data_list

def __split_x_y__(data: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract lob data and annotated label from fi-2010 data
    Parameters
    ----------
    data: Numpy Array
    k: Prediction horizon
    """
    midprice = (data[:,0] + data[:,2])/2
    y = np.zeros(len(midprice) - k)

    for i in range(len(midprice) - k):
        m_i = midprice[i]
        avg_m_j = np.mean(midprice[i+1:i+k+1])
        l_i = avg_m_j / m_i - 1

        if l_i > 0.002:
            y[i] = 1
        elif l_i < -0.002:
            y[i] = 3
        else:
            y[i] = 2

    x = data[:len(midprice) - k, :]
    return x, y

def __data_processing__(x: np.ndarray, y: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Process whole time-series-data
    Parameters
    ----------
    x: Numpy Array of LOB
    y: Numpy Array of annotated label
    T: Length of time frame in single input data
    """
    [N, D] = x.shape

    # x processing
    x_proc = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        x_proc[i - T] = x[i - T:i, :]

    # y processing
    y_proc = y[T - 1:N]
    return x_proc, y_proc

def __load_normalized_data__(filename: str) -> np.ndarray:
    root_path = sys.path[1]
    dataset_path = 'krx'
    file_path = os.path.join(root_path, dataset_path, 'normalized', filename)
    return np.loadtxt(file_path)

class Dataset_krx:
    def __init__(self, normalization: str, tickers: list, days: list, T: int, k: int) -> None:
        """ Initialization """
        self.normalization = normalization
        self.days = days
        self.tickers = tickers
        self.T = T
        self.k = k

        x, y = self.__init_dataset__()
        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

        self.length = len(y)

    def __init_dataset__(self):
        x_cat = np.array([])
        y_cat = np.array([])
        for ticker in self.tickers:
            file_list = get_normalized_data_list(ticker, self.normalization)
            using_file_list = [file_list[i] for i in self.days]
            for file in using_file_list:
                day_data = __load_normalized_data__(file)
                x, y = __split_x_y__(day_data, self.k)
                x_day, y_day = __data_processing__(x, y, self.T)

                if len(x_cat) == 0 and len(y_cat) == 0:
                    x_cat = x_day
                    y_cat = y_day
                else:
                    x_cat = np.concatenate((x_cat, x_day), axis=0)
                    y_cat = np.concatenate((y_cat, y_day), axis=0)

        return x_cat, y_cat

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]