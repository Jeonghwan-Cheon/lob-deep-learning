import os
import sys
import csv
import fnmatch
import numpy as np
from multiprocessing import Pool


def __get_raw__(filename, ticker, compression = 60):
    """
    Handling function for loading raw krx dataset
    Parameters
    ----------
    filename
    ticker: {'KS200', 'KQ150}
    """
    root_path = sys.path[0]
    dataset_path = 'krx'
    dataset_type = 'raw'
    file_path = os.path.join(root_path, dataset_path, dataset_type, filename)

    if ticker == "KS200":
        code = '101SC'
    elif ticker == "KQ150":
        code = '106SC'

    lob_data = []

    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        comp_count = 0
        for idx, row in enumerate(reader):
            if idx > 0 and row[1] == code:
                temp_lob_data = [
                    ###################################
                    #      ask       |       bid      #
                    # price |  vol   |  price |  vol  #
                    ###################################
                    row[18], row[19], row[3],  row[4],  # 1st level
                    row[21], row[22], row[6],  row[7],  # 2nd level
                    row[24], row[25], row[9],  row[10], # 3rd level
                    row[27], row[28], row[12], row[13], # 4th level
                    row[30], row[31], row[15], row[16], # 5th level
                ]

                if len(lob_data) == 0 or temp_lob_data != lob_data[-1]: # avoid duplicated data
                    comp_count= comp_count + 1
                    if comp_count == compression:
                        lob_data.append(temp_lob_data)
                        comp_count = 0

    lob_data = np.array(lob_data).astype(np.float64)
    if ticker == "KS200":
        lob_data[:, list(range(0, 20, 2))] = lob_data[:, list(range(0, 20, 2))] * 100
    elif ticker == "KQ150":
        lob_data[:, list(range(0, 20, 2))] = lob_data[:, list(range(0, 20, 2))] * 10

    return lob_data.astype(np.uint32)


def __single_day_data__(year, month, day, ticker):
    """
    Collect LOB data collected in a single day
    Parameters
    ----------
    year
    month
    day
    ticker: {'KS200', 'KQ150'}
    """
    root_path = sys.path[0]
    dataset_path = 'krx'
    dataset_type = 'raw'
    path1 = str(year)
    filename_type = f"101SC-{str(month)}-{str(day)}-*.csv"

    route = os.path.join(root_path, dataset_path, dataset_type, path1)

    day_file = []
    for file in os.listdir(route):
        if fnmatch.fnmatch(file, filename_type):
            day_file.append(file)
    day_file.sort(key = lambda x:int(x.split('.')[0].split('-')[-1]))

    lob_data_cat = np.array([])

    def get_filename(f):
        return os.path.join(path1, file)

    with Pool() as pool:
        input_files = [(get_filename(file), ticker) for file in day_file]
        lob_data = pool.starmap(__get_raw__, input_files)
        return np.concatenate(lob_data)[1:-1, :]


def __save_preprocessed_data__():
    root_path = sys.path[0]
    dataset_path = 'krx'
    source_path = os.path.join(root_path, dataset_path, 'raw')
    target_path = os.path.join(root_path, dataset_path, 'processed')

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    year_list = [year for year in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, year))]
    day_set = set()

    for year in year_list:
        tmp_day_list = []
        tmp_path = os.path.join(source_path, str(year))
        file_list = [file for file in os.listdir(tmp_path) if file.endswith('.csv')]
        for filename in file_list:
            month = filename.split('-')[1]
            day = filename.split('-')[2]
            tmp_day_info = f'{year}-{month}-{day}'
            day_set.add(tmp_day_info)

    day_list = list(day_set)
    day_list.sort(key = lambda x: (int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])))

    target_ticker = ["KS200", "KQ150"]
    for day_info in day_list:
        for ticker in target_ticker:
            tmp_filename = f'{ticker}-{day_info}.txt'
            tmp_path = os.path.join(target_path, tmp_filename)
            if not os.path.exists(tmp_path):
                tmp_year = int(day_info.split('-')[0])
                tmp_month = int(day_info.split('-')[1])
                tmp_day = int(day_info.split('-')[2])
                processed_data = __single_day_data__(tmp_year, tmp_month, tmp_day, ticker)
                np.savetxt(tmp_path, processed_data, fmt='%.4e')
                print(f"{tmp_filename} saved")
    return

def __get_day_list__():
    __save_preprocessed_data__()
    root_path = sys.path[0]
    dataset_path = 'krx'
    target_path = os.path.join(root_path, dataset_path, 'processed')
    file_list = [file.split('.')[0].split('-', 1)[1] for file in os.listdir(target_path) if file.endswith('.txt')]
    file_list = list(set(file_list))
    file_list.sort(key = lambda x: (int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])))
    return file_list


def __get_norm_info__(path, day, ticker) -> (np.array, np.array):
    filename = f'{ticker}-{day}.txt'
    tmp_path = os.path.join(path, filename)
    data = np.loadtxt(tmp_path)
    price_data = data[:, list(range(0, 20, 2))].flatten()
    volume_data = data[:, list(range(1, 20, 2))].flatten()

    info_price = np.array([len(data),
                        np.mean(price_data), np.std(price_data),
                        np.min(price_data), np.max(price_data),
                        len(str(round(np.max(price_data))))])
    info_volume = np.array([len(data),
                        np.mean(volume_data), np.std(volume_data),
                        np.min(volume_data), np.max(volume_data),
                        len(str(round(np.max(volume_data))))])

    return info_price, info_volume


def __normalize_data__(ticker, normalization, period=1):
    root_path = sys.path[0]
    dataset_path = 'krx'
    source_path = os.path.join(root_path, dataset_path, 'processed')
    target_path = os.path.join(root_path, dataset_path, 'normalized')

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    day_list = __get_day_list__()

    pool = Pool()
    input_list = [(source_path, day, ticker) for day in day_list]
    result = pool.starmap(__get_norm_info__, input_list)

    # index: N, mean, std, min, max, k
    norm_info_price = np.zeros((len(day_list), 6))
    norm_info_volume = np.zeros((len(day_list), 6))

    for idx, (price, volume) in enumerate(result):
        norm_info_price[idx, :] = price
        norm_info_volume[idx, :] = volume

    for idx, day in enumerate(day_list):
        if idx+1 > period:
            source_filename = f'{ticker}-{day}.txt'
            tmp_source_path = os.path.join(source_path, source_filename)
            data = np.loadtxt(tmp_source_path)

            target_filename = f'{ticker}-{day}-{normalization}.txt'
            tmp_target_path = os.path.join(target_path, target_filename)

            if not os.path.exists(tmp_target_path):
                price_minus_term = 0
                price_divide_term = 0
                volume_minus_term = 0
                volume_divide_term = 0

                if normalization == 'Zscore':
                    price_minus_term = np.sum(np.multiply(norm_info_price[idx-1:idx+period-1,0],
                                                          norm_info_price[idx-1:idx+period-1,1])) \
                                       / np.sum(norm_info_price[idx-1:idx+period-1,0]) # price mean
                    price_divide_term = np.sqrt((np.sum(np.multiply(norm_info_price[idx-1:idx+period-1,0],
                                                                    np.square(norm_info_price[idx-1:idx+period-1,2])))
                                                 /np.sum(norm_info_price[idx-1:idx+period-1,0]))) # price std
                    volume_minus_term = np.sum(np.multiply(norm_info_volume[idx - 1:idx + period - 1, 0],
                                                    norm_info_volume[idx - 1:idx + period - 1, 1])) \
                                 / np.sum(norm_info_volume[idx - 1:idx + period - 1, 0]) # volume mean
                    volume_divide_term = np.sqrt((np.sum(np.multiply(norm_info_volume[idx - 1:idx + period - 1, 0],
                                                            np.square(norm_info_volume[idx - 1:idx + period - 1, 2])))
                                         / np.sum(norm_info_volume[idx - 1:idx + period - 1, 0]))) # volume std

                elif normalization == 'MinMax':
                    price_min = np.min(norm_info_price[idx-1:idx+period-1,3])
                    price_max = np.max(norm_info_price[idx-1:idx+period-1,4])
                    volume_min = np.min(norm_info_volume[idx - 1:idx + period - 1, 3])
                    volume_max = np.max(norm_info_volume[idx - 1:idx + period - 1, 4])

                    price_minus_term = price_min
                    price_divide_term = price_max - price_min
                    volume_minus_term = volume_min
                    volume_divide_term = volume_max - volume_min

                elif normalization == 'DecPre':
                    price_decimal = np.max(norm_info_price[idx-1:idx+period-1,5])
                    volume_decimal = np.max(norm_info_volume[idx - 1:idx + period - 1, 5])

                    price_minus_term = 0
                    price_divide_term = 10 ** price_decimal
                    volume_minus_term = 0
                    volume_divide_term = 10 ** volume_decimal

                normalized_data = np.zeros(data.shape)
                normalized_data[:, list(range(0, 20, 2))] = (data[:, list(range(0, 20, 2))] - price_minus_term) / price_divide_term
                normalized_data[:, list(range(1, 20, 2))] = (data[:, list(range(1, 20, 2))] - volume_minus_term) / volume_divide_term

                np.savetxt(tmp_target_path, normalized_data, fmt='%.7e')
                print(f"{target_filename} saved")


def get_normalized_data_list(ticker, normalization):
    __normalize_data__(ticker, normalization)
    root_path = sys.path[0]
    dataset_path = 'krx'
    target_path = os.path.join(root_path, dataset_path, 'normalized')
    file_list = [file for file in os.listdir(target_path)
                 if file.endswith(f'-{normalization}.txt') and file.startswith(f'{ticker}-')]
    file_list = list(set(file_list))
    file_list.sort(key = lambda x: (int(x.split('-')[1]), int(x.split('-')[2]), int(x.split('-')[3])))
    return file_list

def get_processed_data_list(ticker):
    root_path = sys.path[0]
    dataset_path = 'krx'
    target_path = os.path.join(root_path, dataset_path, 'processed')
    file_list = [file for file in os.listdir(target_path) if file.startswith(f'{ticker}-')]
    file_list = list(set(file_list))
    file_list.sort(key = lambda x: (int(x.split('.')[0].split('-')[1]), int(x.split('.')[0].split('-')[2]), int(x.split('.')[0].split('-')[3])))
    return file_list