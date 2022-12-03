from loggers import logger
from optimizers import train
from simulator import evaluate

from loaders.krx_loader import __test_label_dist__, __vis_sample_lob__
from simulator.market_sim import backtest
from simulator.training_vis import vis_training_process


if __name__ == '__main__':
    model_id = logger.generate_id('deeplob_final')
    train.train(
        model_id=model_id, dataset_type = 'fi2010', normalization= 'Zscore', lighten= False,
        T= 100, k= 4, stock= [0, 1, 2, 3, 4], train_test_ratio = 0.7, model_type='deeplob')
    #model_id = 'lobster-lighten_2022-12-03_10:34:05'
    evaluate.test(model_id = model_id, model_type='deeplob')

    # model_id = 'deeplob-lighten_selected_1'
    # vis_training_process(model_id = model_id)
