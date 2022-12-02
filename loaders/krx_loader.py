import multiprocessing
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from loaders.krx_preprocess import get_normalized_data_list, get_processed_data_list


def __split_x_y__(norm_data, proc_data, k, threshold=0.05/100, vis = True):
    """
    Extract lob data and annotated label from fi-2010 data
    Parameters
    ----------
    data: Numpy Array
    k: Prediction horizon
    """
    midprice = (proc_data[:, 0] + proc_data[:, 2]) / 2
    y = np.zeros(len(midprice) - 2*k)

    for i in range(len(midprice) - 2*k):

        m_i = midprice[i+k]
        m_p = np.mean(midprice[i + k + 1:i + 2 * k])
        m_m = np.mean(midprice[i : i + k - 1])
        l_i = m_p / m_m - 1

        if l_i > threshold:
            y[i] = 2
        elif l_i < -threshold:
            y[i] = 0
        else:
            y[i] = 1

    midprice = midprice[k:len(midprice) - k]

    if vis:
        plt.subplots()
        plt.plot(midprice)
        for i in range(len(midprice)):
            if y[i]==1:
                pass
            else:
                if y[i] == 2:
                    color='red'
                elif y[i] == 0:
                    color='blue'
                plt.axvspan(i-0.5, i+0.5, color=color, alpha=0.5)

        plt.show()

    x = norm_data[k:len(norm_data) - k, :]
    print(x.shape, y.shape, midprice.shape)
    return x, y, midprice


def __load_normalized_data__(filename):
    root_path = sys.path[0]
    dataset_path = 'krx'
    file_path = os.path.join(root_path, dataset_path, 'normalized', filename)
    return np.loadtxt(file_path)


def __load_processed_data__(filename):
    root_path = sys.path[0]
    dataset_path = 'krx'
    file_path = os.path.join(root_path, dataset_path, 'processed', filename)
    return np.loadtxt(file_path)


def processing(norm, proc, k, T):
    norm_day_data = __load_normalized_data__(norm)
    proc_day_data = __load_processed_data__(proc)

    x, y, midprice = __split_x_y__(norm_day_data, proc_day_data, k)
    data_val = np.concatenate((np.zeros(T), np.ones(y.size - T)), axis=0)

    return x, y, midprice, data_val


class Dataset_krx:
    def __init__(self, normalization, tickers, days, T, k):
        """ Initialization """
        self.normalization = normalization
        self.days = days
        self.tickers = tickers
        self.T = T
        self.k = k

        self.x, self.y, self.midprice, self.data_val = self.__init_dataset__()
        self.length = np.count_nonzero(self.data_val)
        self.val = np.nonzero(self.data_val)[0]

    def __init_dataset__(self):
        x_cat = np.array([])
        y_cat = np.array([])
        data_val_cat = np.array([])

        for ticker in self.tickers:
            norm_file_list = get_normalized_data_list(ticker, self.normalization)
            using_norm_file_list = [norm_file_list[i] for i in self.days]

            proc_file_list = get_processed_data_list(ticker)
            using_proc_file_list = [proc_file_list[i + 1] for i in self.days]

            pool = multiprocessing.Pool()
            input_files = [(using_norm_file_list[i], using_proc_file_list[i], self.k, self.T) for i in
                           range(len(self.days))]
            result = pool.starmap(processing, input_files)

            for x, y, midprice, data_val in result:
                if len(x_cat) == 0 and len(y_cat) == 0:
                    x_cat = x
                    y_cat = y
                    midprice_cat = midprice
                    data_val_cat = data_val
                else:
                    x_cat = np.concatenate((x_cat, x), axis=0)
                    y_cat = np.concatenate((y_cat, y), axis=0)
                    midprice_cat = np.concatenate((midprice_cat, midprice), axis=0)
                    data_val_cat = np.concatenate((data_val_cat, data_val), axis=0)

        return x_cat, y_cat, midprice_cat, data_val_cat

    def __input_normalization__(self, x_data):
        # price = x_data[:, list(range(0, 20, 2))]
        # vol = x_data[:, list(range(1, 20, 2))]
        # x_data[:, list(range(0, 20, 2))] = (price - np.mean(price))/np.std(price)
        # x_data[:, list(range(1, 20, 2))] = (vol - np.mean(vol))/np.std(vol)
        return (x_data - np.mean(x_data))/np.std(x_data)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        raw_index = self.val[index]
        x_data = self.x[raw_index - self.T:raw_index, :]

        x_data = self.__input_normalization__(x_data)

        y_data = self.y[raw_index]

        x_data = np.expand_dims(x_data, axis=0)
        x_data = torch.from_numpy(x_data)
        y_data = torch.tensor(y_data)
        return x_data, y_data

    def get_midprice(self):
        return self.midprice[self.val]

    def get_class_weights(self):
        label = list(self.y[self.val])
        class_weights = [1 - label.count(0) / len(label),
                      1 - label.count(1) / len(label),
                      1 - label.count(2) / len(label)]
        return class_weights


def __test_label_dist__():
    ticker = 'KQ150'
    k = 50
    normalization = 'Zscore'
    for day in range(12):
        norm_file_list = get_normalized_data_list(ticker, normalization)
        using_norm_file = norm_file_list[day]

        proc_file_list = get_processed_data_list(ticker)
        using_proc_file = proc_file_list[day + 1]

        norm_day_data = __load_normalized_data__(using_norm_file)
        proc_day_data = __load_processed_data__(using_proc_file)

        x, y, _ = __split_x_y__(norm_day_data, proc_day_data, k)
        y = list(y)
        print(f'%% Day: {day}')

        for i in [0, 1, 2]:
            print(f'{i}: {y.count(i)}')


def __vis_sample_lob__():
    import matplotlib.pyplot as plt

    ticker = 'KQ150'
    k = 100
    normalization = 'Zscore'
    day = 1
    for idx in range(100):
        compress = 10

        norm_file_list = get_normalized_data_list(ticker, normalization)
        using_norm_file = norm_file_list[day]

        proc_file_list = get_processed_data_list(ticker)
        using_proc_file = proc_file_list[day + 1]

        norm_day_data = __load_normalized_data__(using_norm_file)
        proc_day_data = __load_processed_data__(using_proc_file)

        x, y = __split_x_y__(norm_day_data, proc_day_data, k)
        sample_shot = np.transpose(x[list(range(0 + idx, 100 * compress + idx, compress))])

        image = np.zeros(sample_shot.shape)
        for i in range(5):
            image[14 - i, :] = sample_shot[4 * i, :]
            image[4 - i, :] = sample_shot[4 * i + 1, :]
            image[15 + i, :] = sample_shot[4 * i + 2, :]
            image[5 + i, :] = sample_shot[4 * i + 3, :]

        plt.imshow(image)
        plt.title('Sample LOB from KRX dataset')
        plt.colorbar()
        plt.show()
