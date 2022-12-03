import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_roc_curve


print(classification_report(all_targets, all_predictions, digits=4))
print(confusion_matrix(all_targets, all_predictions))

plot_roc_curve(clf, X_test, y_test)