import os
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from loggers import logger


def report(model_id):
    with open(os.path.join(logger.find_save_path(model_id), 'prediction.pkl'), 'rb') as f:
        all_midprices, all_targets, all_predictions = pickle.load(f)

    test_acc = accuracy_score(all_targets, all_predictions)
    print(f"Test acc: {test_acc:.4f}")
    print(classification_report(all_targets, all_predictions, digits=4))
    print(confusion_matrix(all_targets, all_predictions))
