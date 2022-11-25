import os
import matplotlib.pyplot as plt
from loggers import logger

def vis_training_process(id):
    training_info = logger.read_log(id, 'training')

    plt.figure(figsize=(15,6))

    plt.subplot(1, 2, 1)
    plt.plot(training_info['train_loss_hist'], label='train loss')
    plt.plot(training_info['val_loss_hist'], label='validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(training_info['train_acc_hist'], label='train acc')
    plt.plot(training_info['val_acc_hist'], label='validation acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    path = logger.find_save_path(id)
    plt.savefig(os.path.join(path, 'training_process.eps'), format='eps')