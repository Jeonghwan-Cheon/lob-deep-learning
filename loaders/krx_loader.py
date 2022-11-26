import os
import sys
import numpy as np
import torch

from loaders.krx_preprocess import get_normalized_data_list

def __split_x_y__(data, k):
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
            y[i] = 0
        elif l_i < -0.002:
            y[i] = 2
        else:
            y[i] = 1

    x = data[:len(midprice) - k, :]
    return x, y

def __data_processing__(x, y, T):
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
    y_proc = y[T - 1:N] - 1
    return x_proc, y_proc

def __load_normalized_data__(filename):
    root_path = sys.path[0]
    dataset_path = 'krx'
    file_path = os.path.join(root_path, dataset_path, 'normalized', filename)
    return np.loadtxt(file_path)

class Dataset_krx:
    def __init__(self, normalization, tickers, days, T, k, compression=10):
        """ Initialization """
        self.normalization = normalization
        self.days = days
        self.tickers = tickers
        self.T = T
        self.k = k
        self.compression = compression

        self.x, self.y, self.data_val = self.__init_dataset__()
        self.length = np.count_nonzero(self.data_val)
        self.val = np.nonzero(self.data_val)[0]

    def __init_dataset__(self):
        x_cat = np.array([])
        y_cat = np.array([])
        data_val_cat = np.array([])

        for ticker in self.tickers:
            file_list = get_normalized_data_list(ticker, self.normalization)
            using_file_list = [file_list[i] for i in self.days]
            for file in using_file_list:
                day_data = __load_normalized_data__(file)
                x, y = __split_x_y__(day_data, self.k)
                data_val = np.concatenate((np.zeros(self.T), np.ones(y.size - self.T)), axis=0)

                if self.compression != 1:
                    comp_length = np.floor(len(y)/self.compression)
                    sampler = list(range(0, int(comp_length * self.compression), self.compression))
                    x = x[sampler]
                    y = y[sampler]
                    data_val = data_val[sampler]

                if len(x_cat) == 0 and len(y_cat) == 0:
                    x_cat = x
                    y_cat = y
                    data_val_cat = data_val
                else:
                    x_cat = np.concatenate((x_cat, x), axis=0)
                    y_cat = np.concatenate((y_cat, y), axis=0)
                    data_val_cat = np.concatenate((data_val_cat, data_val), axis=0)

        return x_cat, y_cat, data_val_cat

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        raw_index = self.val[index]
        x_data = self.x[raw_index-self.T:raw_index, :]
        y_data = self.y[raw_index]

        x_data = np.expand_dims(x_data, axis=0)
        x_data = torch.from_numpy(x_data)
        y_data = torch.tensor(y_data)
        return x_data, y_data

def __test_label_dist__():
    ticker = 'KQ150'
    k = 100
    T = 100
    normalization = 'Zscore'
    day = 0

    file_list = get_normalized_data_list(ticker, normalization)
    file = file_list[day]

    day_data = __load_normalized_data__(file)
    x, y = __split_x_y__(day_data, k)
    y = list(y)

    for i in [0, 1, 2]:
        print(f'{i}: {y.count(i)}')


def __vis_sample_lob__():
    import matplotlib.pyplot as plt

    ticker = 'KQ150'
    k = 100
    normalization = 'Zscore'
    day = 1
    idx = 1000
    compress = 10

    file_list = get_normalized_data_list(ticker, normalization)
    file = file_list[day]

    day_data = __load_normalized_data__(file)
    x, y = __split_x_y__(day_data, k)
    sample_shot = np.transpose(x[list(range(0+idx, 100*compress+idx, compress))])

    image = np.zeros(sample_shot.shape)
    for i in range(5):
        image[14 - i , :] = sample_shot[4 * i, :]
        image[4 - i, :] = sample_shot[4 * i + 1, :]
        image[15 + i, :] = sample_shot[4 * i + 2, :]
        image[5 + i, :] = sample_shot[4 * i + 3, :]

    plt.imshow(image)
    plt.title('Sample LOB from KRX dataset')
    plt.colorbar()
    plt.show()