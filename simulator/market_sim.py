import numpy
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from loaders.krx_loader import Dataset_krx
from loaders.fi2010_loader import Dataset_fi2010
from loggers import logger
from simulator.trading_agent import Trading


def __get_data__(model_id):
    with open(os.path.join(logger.find_save_path(model_id), 'prediction.pkl'), 'rb') as f:
        all_midprices, all_targets, all_predictions = pickle.load(f)
    return all_midprices, all_targets - 1, all_predictions - 1


def backtest(model_id):
    midprice, target, prediction = __get_data__(model_id)
    TradingAgent = Trading()

    patience = 5
    patience_count = 0

    for i in range(len(prediction)):
        if 0 < i < len(prediction) - 1:

            # The end of day data
            if abs(midprice[i+1]-midprice[i])/midprice[i] > 0.001:
                TradingAgent.day_start.append(i+1)
                if TradingAgent.long_inventory > 0:
                    TradingAgent.exit_long(midprice[i])
                if TradingAgent.short_inventory > 0:
                    TradingAgent.exit_short(midprice[i])

            # Inside day data
            else:
                # (1) model predicts stationary
                if prediction[i] == 0:
                    pass

                # (2) model predicts up
                elif prediction[i] == 1:
                    # empty inventory -> long
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.long(midprice[i])
                    # already has long position -> pass
                    elif TradingAgent.long_inventory > 0:
                        pass
                    # has counter-position & patience over -> change position
                    elif TradingAgent.short_inventory > 0:
                        patience_count += 1
                        if patience_count >= patience:
                            TradingAgent.exit_short(midprice[i])
                            TradingAgent.long(midprice[i])
                            patience_count = 0

                # (3) model predicts down
                elif prediction[i] == -1:
                    # empty inventory -> short
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.short(midprice[i])
                    # already has short position
                    elif TradingAgent.short_inventory > 0:
                        pass
                    # has counter-position
                    elif TradingAgent.long_inventory > 0:
                        patience_count += 1
                        if patience_count >= patience:
                            TradingAgent.exit_long(midprice[i])
                            TradingAgent.short(midprice[i])
                            patience_count = 0

        # update balance
        TradingAgent.evaluate_balance(midprice[i])

    # plt.plot(TradingAgent.balance_history/TradingAgent.balance_history[0])
    # plt.plot(TradingAgent.index_history/TradingAgent.index_history[0])
    # y = TradingAgent.position_history

    # for i in range(len(y)):
    #     if y[i] == 0:
    #         pass
    #     else:
    #         if y[i] == 1:
    #             color = 'red' #'#FDB631'
    #         elif y[i] == -1:
    #             color = 'blue' #'#C3C3C3'
    #         plt.axvspan(i - 0.5, i + 0.5, color=color)
    #
    # plt.show()

    print(TradingAgent.balance_history/TradingAgent.balance_history[0])
    return


def vis_label(model_id):
    midprice, target, prediction = __get_data__(model_id)

    plt.subplots()
    plt.plot(list(range(len(midprice))), midprice)
    y = prediction

    for i in range(len(midprice)):
        if y[i] == 0:
            pass
        else:
            if y[i] == 1:
                color = 'red' #'#FDB631'
            elif y[i] == -1:
                color = 'blue' #'#C3C3C3'
            plt.axvspan(i - 0.5, i + 0.5, color=color)

    path = logger.find_save_path(model_id)
    plt.show()
    plt.savefig(os.path.join(path, 'training_process.png'), format='png')
