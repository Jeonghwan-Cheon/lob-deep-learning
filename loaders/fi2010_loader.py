import os
import sys
import numpy as np
import torch


def __get_raw__(auction, normalization, day):
    """
    Handling function for loading raw FI2010 dataset
    Parameters
    ----------
    auction: {True, False}
    normalization: {'Zscore', 'MinMax', 'DecPre'}
    day: {1, 2, ..., 10}
    """

    root_path = sys.path[0]
    dataset_path = 'fi2010'

    if auction:
        path1 = "Auction"
    else:
        path1 = "NoAuction"

    if normalization == 'Zscore':
        tmp_path_1 = '1.'
    elif normalization == 'MinMax':
        tmp_path_1 = '2.'
    elif normalization == 'DecPre':
        tmp_path_1 = '3.'

    tmp_path_2 = f"{path1}_{normalization}"
    path2 = f"{tmp_path_1}{tmp_path_2}"

    if normalization == 'Zscore':
        normalization = 'ZScore'

    if day == 1:
        path3 = tmp_path_2 + '_' + 'Training'
        filename = f"Train_Dst_{path1}_{normalization}_CF_{str(day)}.txt"
    else:
        path3 = tmp_path_2 + '_' + 'Testing'
        day = day - 1
        filename = f"Test_Dst_{path1}_{normalization}_CF_{str(day)}.txt"

    file_path = os.path.join(root_path, dataset_path, path1, path2, path3, filename)
    fi2010_dataset = np.loadtxt(file_path)
    return fi2010_dataset


def __extract_stock__(raw_data, stock_idx):
    """
    Extract specific stock data from raw FI2010 dataset
    Parameters
    ----------
    raw_data: Numpy Array
    stock_idx: {0, 1, 2, 3, 4}
    """
    n_boundaries = 4
    boundaries = np.sort(
        np.argsort(np.abs(np.diff(raw_data[0], prepend=np.inf)))[-n_boundaries - 1:]
    )
    boundaries = np.append(boundaries, [raw_data.shape[1]])
    split_data = tuple(raw_data[:, boundaries[i] : boundaries[i + 1]] for i in range(n_boundaries + 1))
    return split_data[stock_idx]


def __split_x_y__(data, lighten):
    """
    Extract lob data and annotated label from fi-2010 data
    Parameters
    ----------
    data: Numpy Array
    """
    if lighten:
        data_length = 20
    else:
        data_length = 40

    x = data[:data_length, :].T
    y = data[-5:, :].T
    return x, y


def __data_processing__(x, y, T, k):
    """
    Process whole time-series-data
    Parameters
    ----------
    x: Numpy Array of LOB
    y: Numpy Array of annotated label
    T: Length of time frame in single input data
    k: Prediction horizon{0, 1, 2, 3, 4}
    """
    [N, D] = x.shape

    # x processing
    x_proc = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        x_proc[i - T] = x[i - T:i, :]

    # y processing
    y_proc = y[T - 1:N]
    y_proc = y_proc[:, k] - 1
    return x_proc, y_proc


class Dataset_fi2010:
    def __init__(self, auction, normalization, stock_idx, days, T, k, lighten):
        """ Initialization """
        self.auction = auction
        self.normalization = normalization
        self.days = days
        self.stock_idx = stock_idx
        self.T = T
        self.k = k
        self.lighten = lighten

        x, y = self.__init_dataset__()
        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

        self.length = len(y)

    def __init_dataset__(self):
        x_cat = np.array([])
        y_cat = np.array([])
        for stock in self.stock_idx:
            for day in self.days:
                day_data = __extract_stock__(
                    __get_raw__(auction=self.auction, normalization=self.normalization, day=day), stock)
                x, y = __split_x_y__(day_data, self.lighten)
                x_day, y_day = __data_processing__(x, y, self.T, self.k)

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


def __vis_sample_lob__():
    import matplotlib.pyplot as plt

    stock = 0
    k = 100
    normalization = 'Zscore'
    day = 9
    idx = 1000
    lighten = True

    day_data = __extract_stock__(
        __get_raw__(auction=False, normalization=normalization, day=day), stock)
    x, y = __split_x_y__(day_data, lighten)
    sample_shot = np.transpose(x[0 + idx:100 + idx])

    image = np.zeros(sample_shot.shape)
    for i in range(5):
        image[14 - i , :] = sample_shot[4 * i, :]
        image[4 - i, :] = sample_shot[4 * i + 1, :]
        image[15 + i, :] = sample_shot[4 * i + 2, :]
        image[5 + i, :] = sample_shot[4 * i + 3, :]

    plt.imshow(image)
    plt.title('Sample LOB from FI-2010 dataset')
    plt.colorbar()
    plt.show()
