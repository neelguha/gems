import numpy as np 
import os 

def load_data(dir):
    """ Loads train/test/val data from directory containing data as *.npy files  

    # Arguments
        dir: directory containing train/test/val *.npy files
    
    # Returns: 
        Xtr, ytr, Xval, yval, Xts, yts 
    """

    Xtr = np.load(os.path.join(dir, "xtr.npy"))
    ytr = np.load(os.path.join(dir, "ytr.npy"))
    Xval = np.load(os.path.join(dir, "xval.npy"))
    yval = np.load(os.path.join(dir, "yval.npy"))
    Xts = np.load(os.path.join(dir, "xts.npy"))
    yts = np.load(os.path.join(dir, "yts.npy"))

    return Xtr, ytr, Xval, yval, Xts, yts 

def flatten(X, y):
    """ flattens X, y across all tasks """
    num_tasks = len(X)
    X_flt = []
    y_flt = []
    for t in range(num_tasks):
        X_flt.extend(X[t])
        y_flt.extend(y[t])
    return np.array(X_flt), np.array(y_flt)