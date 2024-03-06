import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import roc_curve,roc_auc_score

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
   
def ROC(y_test,y_pred):
    fpr,tpr,tr=roc_curve(y_test,y_pred, drop_intermediate=True)
    auc=roc_auc_score(y_test,y_pred)
    idx=np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()
    idx = np.argmax(2 * tpr * (1-fpr) / (tpr + 1-fpr))

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr,tpr,label="TPR(AUC="+str(auc)+")")
    plt.plot(fpr,1-fpr,label="TNR")

    plt.plot(fpr, (2 * tpr * (1-fpr) / (tpr + 1-fpr)), label="TPR x TNR", color='red', linestyle='dashed')
    plt.plot(fpr[idx], (2 * tpr * (1-fpr) / (tpr + 1-fpr))[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]
        