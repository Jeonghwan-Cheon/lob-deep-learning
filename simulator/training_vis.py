import os
import pickle
import matplotlib.pyplot as plt

from loggers import logger


def vis_training_process(model_id):
    with open(os.path.join(logger.find_save_path(model_id), 'training_process.pkl'), 'rb') as f:
        training_info = pickle.load(f)

    plt.figure(figsize=(15,6))

    plt.subplot(1, 2, 1)
    plt.plot(training_info['train_loss_hist'], label='train loss')
    plt.plot(training_info['val_loss_hist'], label='validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0.6, 1.2])

    plt.subplot(1, 2, 2)
    plt.plot(training_info['train_acc_hist'], label='train acc')
    plt.plot(training_info['val_acc_hist'], label='validation acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 0.9])

    path = logger.find_save_path(model_id)
    plt.savefig(os.path.join(path, 'training_process.svg'), format='svg')
