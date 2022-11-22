from optimizers import train

if __name__ == '__main__':
    train.train(
        dataset_type = 'krx', normalization= 'Zscore', lighten= True,
        T= 100, k= 100, stock= ['KS200', 'KQ150'], train_test_ratio = 0.7)