from loggers import logger
from optimizers import train
from simulator import evaluate

from loaders.krx_loader import __test_label_dist__, __vis_sample_lob__


if __name__ == '__main__':
    model_id = logger.generate_id('deeplob-lighten')
    train.train(
        model_id=model_id, dataset_type = 'krx', normalization= 'Zscore', lighten= True,
        T= 100, k= 10, stock= ["KQ150"], train_test_ratio = 0.7)
    evaluate.test(model_id=model_id)
