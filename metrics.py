
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def __metrics(y_test, y_pred):
    accpred = y_pred
    accpred[accpred > 0.5] = 1
    accpred[accpred <= 0.5] = 0
    # print(y_test)
    # print(accpred)

    accuracy = accuracy_score(y_test, accpred)
    print("accuray: ", accuracy)
    precision = precision_score(y_test, y_pred, average='weighted')
    print("precision: ", precision)
    recall = recall_score(y_test, y_pred, average='weighted')
    print("recall: ", recall)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("f1: ", f1)
    auc = roc_auc_score(y_test, y_pred, average='weighted')
    print("auc: ", auc)
    return {"accuray": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}
