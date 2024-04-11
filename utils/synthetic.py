import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm import trange, tqdm
from .constants import bar_format


def teleporting_mouse(f, X):

    x = np.arange(X.shape[1])
    Y = np.zeros(X.shape)

    for i in trange(X.shape[0], bar_format=bar_format):
        y = np.cos(2 * np.pi * f * x + np.pi / X.shape[0] * i)
        #     y = minmax(y)
        y[y < 0] /= 7
        Y[i, :] = y
    Y = softmax(Y, axis=1)

    X_ = np.zeros_like(X)
    neurids = np.arange(X.shape[1])
    for i in trange(X.shape[0], bar_format=bar_format):
        for j in range(int(X[i].sum())):
            idx = np.random.choice(neurids, p=Y[i, :])
            X_[i, idx] = 1

    return x, y, X_, Y


def gaussian(x_values, mu, sig):
    return np.exp(-np.power(x_values - mu, 2.) / (2 * np.power(sig, 2.)))


def minmax(x):
    return (x - min(x)) / (max(x) - min(x))


def scale(x, Max):
    x = minmax(x)
    x *= Max
    return np.round(x).astype('int')


def sort_with_behavioral(X, dat1, params):
    df = pd.DataFrame({'x': np.arange(len(dat1)).tolist(), 'y': scale(dat1, params.N)})
    df['m'] = df.y.rolling(100, center=True).mean()
    df['m'] = df.m.astype(float).apply(np.round)
    sorting_idx = []
    for i in range(params.N):
        sorting_idx.append(X[:, df.m == i].sum(axis=1).argmax())
    return X[sorting_idx, :], sorting_idx


def get_background(X):
    """ takes the CA1 data and permutes everything """
    X_ = np.zeros_like(X)

    # shuffle ISI, i.e. in the time dimension
    for i in range(X.shape[0]):
        spts = np.where(X[i, :] == 1)[0]
        spts = np.insert(spts, 0, 0)
        isi = np.diff(spts)
        np.random.shuffle(isi)
        spts_ = np.cumsum(isi)
        X_[i, spts_] = 1

    # shuffle in the neuronid dimension
    permidx = np.arange(X_.shape[0])
    np.random.shuffle(permidx)
    X_ = X_[permidx, :]
    return X_


def linear_track_back_and_forth(X, f, tuning_curve_std):

    N = X.shape[0]
    x = np.arange(X.shape[1])
    X_ = np.zeros(X.shape)
    mouse_positions = scale(np.cos(2 * np.pi * f * x + np.pi / N), N)
    x_values = np.arange(N)

    for t, pos in enumerate(tqdm(mouse_positions)):
        # firing_probs = softmax(gaussian(x_values, pos, tuning_curve_std)) # NOTE: this samping sucks
        # n = np.random.choice(params.N, p=firing_probs)
        n = np.random.randn() * tuning_curve_std + pos
        n = np.round(np.clip(n, 0, N - 1)).astype(int)
        X_[n, t] = 1

    hf = np.where(X_.sum(axis=1) > 1000)[0]

    for i in hf:
        ons = X_[i, :] == 1
        X_[i, ons] = np.random.choice([0, 1], p=[0.8, 0.2], size=(sum(ons),))  # fix high FR at boundaries
    return X_, x_values, mouse_positions