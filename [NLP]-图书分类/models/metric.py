from sklearn import metrics


def evaluate(y_train, y_test, pred_train, pred_test):
    return metrics.accuracy_score(y_train, pred_train), \
           metrics.accuracy_score(y_test, pred_test), \
           metrics.recall_score(y_test, pred_test, average='micro'), \
           metrics.f1_score(y_test, pred_test, average='weighted')