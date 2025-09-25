import numpy as np

def rmse(obs, sim):
    obs = np.asarray(obs); sim = np.asarray(sim)
    return float(np.sqrt(np.mean((sim - obs)**2)))

def mae(obs, sim):
    obs = np.asarray(obs); sim = np.asarray(sim)
    return float(np.mean(np.abs(sim - obs)))

def nse(obs, sim):
    obs = np.asarray(obs); sim = np.asarray(sim)
    return float(1.0 - np.sum((sim-obs)**2) / np.sum((obs - np.mean(obs))**2))

def kge(obs, sim):
    obs = np.asarray(obs); sim = np.asarray(sim)
    r = np.corrcoef(obs, sim)[0,1]
    alpha = np.std(sim)/np.std(obs)
    beta  = np.mean(sim)/np.mean(obs)
    return float(1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))
