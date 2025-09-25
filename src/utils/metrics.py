import numpy as np

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)))

def nse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

def kge(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r = np.corrcoef(y_true, y_pred)[0,1]
    alpha = np.std(y_pred)/np.std(y_true)
    beta = np.mean(y_pred)/np.mean(y_true)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
