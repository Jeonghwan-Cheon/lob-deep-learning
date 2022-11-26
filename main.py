from loggers import logger
from optimizers import train
from simulator import evaluate

if __name__ == '__main__':

    model_id = logger.generate_id('deeplob-lighten')
    train.train(
        model_id=model_id, dataset_type = 'fi2010', normalization= 'Zscore', lighten= False,
        T= 100, k= 4, stock= [0, 1, 2, 3, 4], train_test_ratio = 0.7)
    evaluate.test(model_id=model_id)
