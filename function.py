import numpy as np

def cos_sin_real_sol(x):
    return [2*np.exp(x) - 1/np.exp(x), -np.cos(x) + 2*np.exp(x) + 1/np.exp(x)]

def cos_sin_f1(x, y):
    return np.cos(x) + y

def cos_sin_f2(x, y):
    return np.sin(x) + y
