from loggers import logger
from optimizers import train
from simulator import evaluate, market_sim, training_vis, classification_report

if __name__ == '__main__':
    # experiment parameter setting
    dataset_type = 'fi2010'
    normalization = 'Zscore'
    lighten = True
    model_type = 'lobster'

    T = 100
    k = 4
    stock = [0, 1, 2, 3, 4]
    train_test_ratio = 0.7

    # generate model id
    model_id = logger.generate_id(model_type)
    print(f"Model ID: {model_id}")

    train.train(model_id=model_id, dataset_type=dataset_type, normalization=normalization,
                lighten=True,T=T, k=k, stock=stock, train_test_ratio=train_test_ratio,
                model_type=model_type)
    evaluate.test(model_id=model_id, model_type=model_type)
    classification_report.report(model_id=model_id)
    training_vis.vis_training_process(model_id=model_id)
    if dataset_type == "KRX":
        market_sim.backtest(model_id=model_id)

