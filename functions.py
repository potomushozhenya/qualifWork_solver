import numpy as np
from dotenv import load_dotenv as load

k1 = load('k1')
# cos_sin
def cos_sin_real_sol(x):
    return [2*np.exp(x) - 1/np.exp(x), -np.cos(x) + 2*np.exp(x) + 1/np.exp(x)]

def cos_sin_f1(x, y):
    return np.cos(x) + y

def cos_sin_f2(x, y):
    return np.sin(x) + y

# simple
def simple_real_sol(x):
    return [np.cos(x), np.sin(x)]

def simple_f1(x, y):
    return -y

def simple_f2(x, y):
    return y

# spring pendulum
def spring_pendulum_f1(x, y):
    return y

def spring_pendulum_f2(x, y):
    global k1
    return -k1 * np.sin(x)

def spring_pendulum_real_sol(x):
    global k1
    return [k1 * np.sin(x) + 0.8 * x + 1, k1 * np.cos(x) + 0.8]