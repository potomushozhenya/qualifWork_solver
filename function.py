import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()
k1 = float(os.getenv('k1'))

# cos_sin
def cos_sin_real_sol(x):
    return [2*np.exp(x) - 1/np.exp(x), -np.cos(x) + 2*np.exp(x) + 1/np.exp(x)]

def cos_sin_f1(x, y):
    return np.cos(x) + y

def cos_sin_f2(x, y):
    return np.sin(x) + y

# simple
def simple_real_sol(x):
    return [np.cos(x) - np.sin(x), np.sin(x) + np.cos(x)]

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

# test x argument
def test_x_f1(x, y):
    return x

def test_x_f2(x, y):
    return x

def test_x_real_sol(x):
    return [x*x/2 + 1, x*x/2 + 1]

# cos

def cos_f1(x, y):
    return np.cos(x) * y

def cos_f2(x, y):
    return np.cos(x) * y

def cos_real_sol(x):
    return [np.exp(np.sin(x)) + 2 * np.exp(-np.sin(x)), np.exp(np.sin(x)) - 2 * np.exp(-np.sin(x))]

def stiff_f1(x, y):
    return -50 * (y - np.cos(x))

def stiff_f2(x, y):
    return -50 * (y - np.cos(x))

def stiff_real_sol(x):
    return [(50*np.sin(x))/2501 + (2500*np.cos(x))/2501 + 1/(2501*np.exp(50*x)), (50*np.sin(x))/2501 + (2500*np.cos(x))/2501 + 1/(2501*np.exp(50*x))]

def stiff_v2_f2(x, y):
    return 50 * (y - np.cos(x))

def stiff_v2_real_sol(x):
    return [(1/2499)*np.sin(50*x) + (-1/2499)*np.cos(50*x) - 50/2499*np.sin(x) + 2500/2499 * np.cos(x),
            (-1/2499)*np.sin(50*x) - (1/2499)*np.cos(50*x) + 50/2499*np.sin(x) + 2500/2499 * np.cos(x)]