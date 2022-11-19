import os
import sys
import csv
import fnmatch
import numpy as np
import torch

def __get_raw__(filename: str, ticker: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Handling function for loading raw krx dataset
    Parameters
    ----------
    filename
    ticker: {'KS200', 'KQ150}
    """
    root_path = sys.path[1]
    dataset_path = 'krx'
    file_path = os.path.join(root_path, dataset_path, filename)

    if ticker == "KS200":
        code = '101SC'
    elif ticker == "KQ150":
        code = '106SC'

    lob_data = []
    mid_price = []

    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx > 0 and row[1] == code:
                temp_lob_data = [
                    #################################
                    #      ask      |      bid      #
                    # price |  vol  | price |  vol  #
                    #################################
                    row[3], row[4], row[18], row[19],   # 1st level
                    row[6], row[7], row[21], row[22],   # 2nd level
                    row[9], row[10], row[24], row[25],  # 3rd level
                    row[12], row[13], row[27], row[28], # 4th level
                    row[15], row[16], row[30], row[31]  # 5th level
                ]

                temp_mid_price = (float(row[3]) + float(row[18])) / 2

                if len(lob_data) == 0 or temp_lob_data != lob_data[-1]:
                    lob_data.append(temp_lob_data) # avoid duplicated data
                    mid_price.append(temp_mid_price)

    return np.array(lob_data), np.array(mid_price)

def __single_day_data__(year: int, month: int, day: int, ticker: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect LOB data collected in a single day
    Parameters
    ----------
    year
    month
    day
    ticker: {'KS200', 'KQ150}
    """
    root_path = sys.path[1]
    dataset_path = 'krx'
    path1 = str(year)
    filename_type = f"101SC-{str(month)}-{str(day)}-*.csv"

    route = os.path.join(root_path, dataset_path, path1)

    day_file = []
    for file in os.listdir(route):
        if fnmatch.fnmatch(file, filename_type):
            day_file.append(file)
    day_file.sort()

    lob_data_cat = np.array([])
    mid_price_cat = np.array([])

    for file in day_file:
        filename = os.path.join(path1, file)
        lob_data_piece, mid_price_piece = __get_raw__(filename, ticker)

        if len(lob_data_cat) == 0 and len(mid_price_cat) == 0:
            lob_data_cat = lob_data_piece
            mid_price_cat = mid_price_piece
        else:
            lob_data_cat = np.concatenate((lob_data_cat, lob_data_piece), axis=0)
            mid_price_cat = np.concatenate((mid_price_cat, mid_price_piece), axis=0)

    return lob_data_cat, mid_price_cat


class Dataset_krx:
    def __init__(self, normalization: str, tickers: list, days: list, T: int, k: int, lighten: bool) -> None:
        """ Initialization """
        self.normalization = normalization
        self.days = days
        self.stock_idx = tickers
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
