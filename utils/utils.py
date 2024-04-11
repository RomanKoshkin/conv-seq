from itertools import chain, groupby
from collections import Counter
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical

import sys, time, json, yaml, subprocess, copy
from tqdm import trange
from utils.constants import bar_format

import os, pickle, shutil, warnings, shutil
import numpy as np
from multiprocessing import Pool
from scipy import signal
import numba
from numba import njit, jit, prange
from numba.typed import List

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from termcolor import cprint
from pprint import pprint

import networkx as nx
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph
from collections import namedtuple
# from sknetwork.clustering import Louvain, modularity

from utils.nulldist import NullDistEstimator
from utils.metrics import get_metrics


class AttributeDict(dict):
    """ convenience class. To get and set properties both as if it were a dict or obj """
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def load_config(pathAndNameOfConfig):
    yaml_txt = open(pathAndNameOfConfig).read()
    parms_dict = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    parms_obj = AttributeDict()
    for k, v in parms_dict.items():
        parms_obj[k] = v
    return parms_obj


def diff(x, dim=0):
    if dim == 1:
        return x[:, 1:] - x[:, :-1]
    else:
        raise NotImplementedError('Not implemented')


def highlight_above_q99(x, q_99, ax, xlim, highlight=True):
    A = []
    started = False

    for i in range(x.shape[0]):

        if not started:
            if x[i] > q_99:
                st = i
                started = True
        if started:
            if x[i] < q_99:
                en = i
                A.append((st, en - 1))
                started = False

    for it in A:
        if it[1] - it[0] >= 10:
            mid = np.round(np.mean(it)).astype(int)
            if mid < xlim[1]:
                if highlight:
                    ax.axvspan(*it, color='red', alpha=0.4)
                ax.plot(mid, x[mid] + 5, "k*")

    return A


class Plotter(object):

    def __init__(self, model, params, X, Xtorch, dat1, loadnull=False):

        self.Xtorch = Xtorch
        self.model = model
        self.params = params
        self.dat1 = dat1
        self.X = X

        # get "null" projections for each filter on init
        OUT, FILT = self.model.forward(self.Xtorch)
        self.proj0 = []
        for out in OUT:
            proj = out.cpu().detach().numpy().squeeze()
            self.proj0.append(np.copy(proj))

        self.estimator = NullDistEstimator(X, params)
        self.estimator.estimate(loadnull=loadnull)

    def plot(self, ep, loss_dict=None, close=True):

        OUT, FILT = self.model.forward(self.Xtorch)

        for k, (out, filt, proj0) in enumerate(zip(OUT, FILT, self.proj0)):
            out = out.squeeze().detach().cpu().flatten().numpy()
            f = filt.squeeze().detach().cpu().numpy()

            srt = sorted(list(zip(range(self.params.N), f.argmax(axis=1))), key=lambda tup: tup[1], reverse=False)
            newids = [i[0] for i in srt]

            gs_kw = dict(height_ratios=[5, 1])
            fig, ax = plt.subplots(2, 4, figsize=(15, 4), sharex=False, gridspec_kw=gs_kw)
            plt.suptitle(f'Epoch {ep}', fontsize=22)
            ax[0, 0].imshow(f, aspect='auto')
            ax[0, 1].imshow(f[newids, :], aspect='auto')
            ax[0, 0].set_yticks([])
            ax[0, 1].set_yticks([])
            ax[0, 0].set_ylabel('Neuron IDs')
            ax[1, 0].hist(self.model.mus[k].detach().cpu().numpy().squeeze(), bins=20)
            ax[1, 1].hist(self.model.mus[k].detach().cpu().numpy().squeeze(), bins=20)
            [ax[i, j].set_xlim(0, self.params.W) for i in range(2) for j in range(2)]

            ax[1, 0].spines['right'].set_visible(False)
            ax[1, 0].spines['top'].set_visible(False)
            ax[1, 0].set_yticks([])
            ax[1, 0].set_ylabel('Dist')
            ax[1, 1].spines['right'].set_visible(False)
            ax[1, 1].spines['top'].set_visible(False)
            ax[1, 1].set_yticks([])
            #     ax[1,1].set_ylabel('Dist')

            # ax[0, 2].hist(proj0, bins=50, color='blue', alpha=0.4)
            # ax[0, 2].hist(out, bins=50, color='red', alpha=0.4)

            ax[0, 2].hist(self.estimator.nulldist_full, bins=50, color='blue', alpha=0.4)
            for q in (self.estimator.q_1, self.estimator.q_99):
                ax[0, 2].axvline(q)
            ax[0, 2].twinx().hist(out, bins=50, color='red', alpha=0.4)

            ax[0, 3].hist(self.estimator.nulldist, bins=50, color='blue', alpha=0.4)
            for q in np.quantile(self.estimator.nulldist, (0.001, 0.999)):
                ax[0, 3].axvline(q)
            ax[0, 3].axvline(out.sum(), color='red')
            plt.savefig(f'{self.params.path_to_data}/a{k}_{ep:05d}.jpeg')
            if close:
                plt.close('all')

            fig = plt.figure(figsize=(17, 8), constrained_layout=False)
            plt.suptitle(f'Epoch {ep}', fontsize=22)
            spec = fig.add_gridspec(ncols=20, nrows=6)
            ax0 = fig.add_subplot(spec[:4, :14])
            ax0.spy(self.X[newids, :], aspect='auto', markersize=2, color='k')
            ax0.set_xlim(0, self.params.xlim)
            ax0.set_yticks([])
            ax0.set_xticks([])
            ax0.set_ylabel('Neuron IDs')

            ax1 = fig.add_subplot(spec[4:, :14])
            ax1.set_yticks([])
            # ax1.set_xticks([])
            ax1.plot(
                out,
                label=f'$k={k}$',
            )
            if self.params.K > 1:
                if k == 0:
                    ax1.plot(
                        OUT[1].squeeze().detach().cpu().numpy(),
                        color='gray',
                        alpha=0.5,
                        label=f'$k=1$',
                    )
                else:
                    ax1.plot(
                        OUT[0].squeeze().detach().cpu().numpy(),
                        color='gray',
                        alpha=0.5,
                        label=f'$k=2$',
                    )
                ax1.legend(loc='upper right')
            axt = ax1.twinx()
            axt.set_yticks([])
            # axt.plot(self.dat1, 'r', lw=1)
            ax1.axhline(self.estimator.q_99, lw=1, color='k', ls=":")
            ax1.set_ylim(0, 1.2 * self.estimator.q_99)
            ax1.set_xlim(0, self.params.xlim)
            ax1.set_ylabel('$\mathbf{X}*\mathbf{W}^{(k)}$')
            ax1.set_xlabel('Time steps')

            if loss_dict is not None:
                ax2 = fig.add_subplot(spec[:3, 14:17])
                ax2.set_yticks([])
                ax2.set_xticks([])
                ax2.plot(loss_dict['proj'], label='$\sum{X*C}$')
                ax2.legend(fontsize=10)

                ax3 = fig.add_subplot(spec[3:, 14:17])
                ax3.set_yticks([])
                ax3.set_xticks([])
                ax3.plot(loss_dict['proj_ent'], label='$H(X*C)$')
                ax3.legend(fontsize=10)

                ax4 = fig.add_subplot(spec[:3, 17:20])
                ax4.set_yticks([])
                ax4.set_xticks([])
                ax4.plot(loss_dict['TV'], label='$TV(X*C)$')
                ax4.legend(fontsize=10)

                ax5 = fig.add_subplot(spec[3:, 17:20])
                ax5.set_yticks([])
                ax5.set_xticks([])
                ax5.plot(loss_dict['mu_ent'], label='$H(\mu)$')
                ax5.legend(fontsize=10)

            if close:
                plt.savefig(f'{self.params.path_to_data}/b{k}_{ep:05d}.jpeg')
                plt.close('all')
        return OUT


class Trainer(object):

    def __init__(self, model, optimizer, dat1, X, GT, params, loadnull=False):
        self.model = model.to(params.device)
        self.optimizer = optimizer
        self.dat1 = dat1
        self.X = X
        self.GT = GT  # list of lists, each (int) mid time steps of true sequences
        self.params = params
        self.Xtorch = torch.from_numpy(X).view(
            1,
            1,
            params.N,
            params.Ts,
        ).float().to(params.device)
        self.cossim = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.KLD = torch.nn.KLDivLoss(reduction='sum')
        self.plotter = Plotter(model, params, X, self.Xtorch, dat1, loadnull=loadnull)
        self.loss_dict = dict(
            ep=[],
            proj=[],
            proj_ent=[],
            TV=[],
            mu_ent=[],
            filt_xcor=[],
            proj_sim=[],
        )
        self.EP = 0
        self.metrics = []
        self.smoothing_ker = torch.ones(
            size=(
                1,
                1,
                1,
                self.params.kerwidth,
            ),
            device=params.device,
        ) / self.params.kerwidth
        torch.save(model.state_dict(), '../checkpoints/untrained.torch')

    def JSD(self, OUT):
        """ Jensen-Shannon divergence between a list of vectors """
        Djs_ACC = torch.tensor([0.0], dtype=torch.float, device=self.params.device)
        for i_ in range(len(OUT)):
            for j_ in range(len(OUT)):
                if j_ < i_:
                    p = torch.softmax(OUT[i_].squeeze(), dim=0)
                    q = torch.softmax(OUT[j_].squeeze(), dim=0)
                    m = 0.5 * (p + q)
                    Dpm = self.KLD(m.log(), p)
                    Dqm = self.KLD(m.log(), q)
                    Djs = 0.5 * Dpm + 0.5 * Dqm
                    Djs_ACC -= Djs
        return Djs_ACC

    def proj_xcorr(self, a, b):
        a_ = a.view(1, 1, -1) - a.mean()
        b_ = b.view(1, 1, -1) - b.mean()
        l = len(a.squeeze())
        s0 = a_.std()
        s1 = b_.std()
        return (F.conv1d(a_, b_, padding=self.params.xcorr_padding) / (s0 * s1) / l).mean()

    def cosine_similarity(self, OUT):
        c = 0
        sim = torch.tensor([0], device=self.params.device, dtype=torch.float)
        for i, out0 in enumerate(OUT):
            for j, out1 in enumerate(OUT):
                if j > i:  # NOTE: only take the upper triangle, excluding the main diag
                    if self.params.cossim:
                        out0 = F.conv2d(out0, self.smoothing_ker, padding=(0, 100))
                        out1 = F.conv2d(out1, self.smoothing_ker, padding=(0, 100))
                        sim += self.cossim(
                            out0.squeeze().flatten().pow(self.params.simpow),
                            out1.squeeze().flatten().pow(self.params.simpow),
                            # out0.squeeze().flatten(),
                            # out1.squeeze().flatten(),
                        )
                    proj_xc = self.proj_xcorr(out0, out1).clamp(min=0.02)
                    c += 1
        return sim / c + 0.8 * proj_xc  # NOTE: normalize

    # FIXME: WIP
    def xcorr_similarity(self, FILT):
        c = 0
        xcorr = torch.tensor([0], device=self.params.device, dtype=torch.float)
        for i in range(len(FILT)):
            for j in range(len(FILT)):
                if j > i:  # NOTE: only take
                    xcorr += F.conv2d(FILT[i], FILT[j], padding=(0, self.params.W)).sum()
                    c += 1
        return xcorr / c  # NOTE: normalize

    def train(self, epochs=200, loadnull=False):
        pbar = trange(self.EP, epochs, desc=f'{0:.3f}\t{0:.3f}\t{0:.3f}\t{0:.3f}\t{0:.2f}', bar_format=bar_format)
        for i in pbar:
            try:
                tt = time.time()
                self.optimizer.zero_grad()
                OUT, FILT = self.model.forward(self.Xtorch)

                loss = torch.tensor([0], device=self.params.device, dtype=torch.float)

                for k, (out, filt) in enumerate(zip(OUT, FILT)):
                    reconst = out.pow(2).sum()
                    entropy = Categorical(torch.softmax(
                        out.view(
                            self.params.out_channels,
                            -1,
                        ),
                        dim=1,
                    )).entropy().sum()
                    entropyOfMus = Categorical(
                        torch.softmax(
                            self.model.mus[k].view(
                                self.params.out_channels,
                                -1,
                            ),
                            dim=1,
                        )).entropy().sum()
                    # TV = diff(out.view(out_channels, -1), dim=1).pow(2).sum() # penalize the energy of the derivative ()
                    TV = diff(
                        out.view(
                            self.params.out_channels,
                            -1,
                        ),
                        dim=1,
                    ).abs().sum()  # penalize the energy of the derivative ()

                    loss += -self.params.proj_l * reconst  # maximize total filter response
                    loss += self.params.proj_ent_l * entropy  # minimize proj entropy
                    loss += self.params.TV_l * TV  # minimize proj total variation
                    loss += -self.params.filt_ent_l * entropyOfMus  # maximize filt entropy

                # add filter similarity loss

                if len(OUT) > 1:
                    sim_cos = self.cosine_similarity(OUT)
                    # sim_cos = self.JSD(OUT)  # NOTE: WIP
                    sim_xcor = self.xcorr_similarity(FILT)

                    loss += self.params.proj_sim_l * sim_cos  # minimize projection similarity
                    loss += self.params.filt_sim_l * sim_xcor  # minimize filter similarity
                else:
                    sim_cos = torch.tensor([0])
                    sim_xcor = torch.tensor([0])

                # FIXME: return a dict of losses
                self.loss_dict['ep'].append(i)
                self.loss_dict['proj'].append(reconst.item())
                self.loss_dict['proj_ent'].append(entropy.item())
                self.loss_dict['TV'].append(TV.item())
                self.loss_dict['mu_ent'].append(entropyOfMus.item())
                self.loss_dict['filt_xcor'].append(sim_xcor.item())
                self.loss_dict['proj_sim'].append(sim_cos.item())

                loss.backward(retain_graph=True)
                self.optimizer.step()
                elapsed_s = time.time() - tt
                if self.params.snapshot_interval is not None:
                    if i % self.params.snapshot_interval == 0:
                        proj = self.plotter.plot(i, self.loss_dict)
                # mus.clamp_(min=1, max=99)

                true_discovery, false_discovery, non_discovery = get_metrics(
                    OUT[0].squeeze().detach().cpu().numpy(),
                    self.GT[0],
                    self.plotter.estimator.q_99,
                    self.params,
                    debug=False,
                )
                self.metrics.append({
                    "ep": self.EP,
                    "TD": true_discovery,
                    "FD": false_discovery,
                    "ND": non_discovery,
                    "x": reconst.item() / self.params.Ts,
                })

                pbar.set_description(
                    f'proj: {reconst.item()/100000:.2f} | H(proj):{entropy.item():.2f} | TV: {TV.item()/1000:.2f} | H(mu): {entropyOfMus.item():.2f} | fxcor: {sim_xcor.item():.2f} | proj_sim: {sim_cos.item():.2f} | TD: {true_discovery} | FD: {false_discovery} | ND: {non_discovery}'
                )

                # early stopping:
                if true_discovery / (true_discovery + false_discovery + non_discovery) == 1:
                    cprint('Early termnination', color='yellow')
                    break

                self.EP += 1
            except KeyboardInterrupt:
                cprint('User interrupt', color='green')
                break


def tonp(x):
    return x.cpu().detach().numpy()


def get_significants(ids, Z, class_id):
    # get span lengths get continous spans of class labels
    idx0 = np.where((ids == class_id).cpu().numpy())[0]
    idx0_spans = get_continuous_indices(np.diff(idx0))

    # get span lengths
    for i, span in enumerate(idx0_spans):
        idx0_spans[i] += (span[1] - span[0],)

    # sort by width, widest will come first
    idx0_spans = sorted(idx0_spans, key=lambda tup: tup[2], reverse=True)

    span_lengths = np.array([i[2] for i in idx0_spans])
    median = np.median(span_lengths)
    median_len_span_idx = np.argmin(span_lengths - median)

    # span = idx0_spans[median_len_span_idx] # take the span with the median width as a reference
    span = idx0_spans[0]  # take the span with the largets width as a reference
    st, en = span[0], span[1]
    significants = np.where(Z[idx0[st]:idx0[en], :].mean(0) > 0.01)[0]  # at least half of the points
    return significants


def getLimitsOfContigElements(arr, highest_threshold):
    """
    RETURNS:
        start and stop indices of elements that are lower than the threshold
    """
    org_idx = np.where(arr < highest_threshold)[0]  # get indices of items in the orig array that satisfy the condition
    diffs = np.diff(org_idx)  # get diffs
    lims = get_continuous_indices(diffs)  # get indices of consecutive items that satisfy the condition
    return [(org_idx[st], org_idx[en]) for st, en in lims]


def get_continuous_indices(arr):
    """ 
    ARGS:
        np array of int or bools
        e.g. 
            arr = np.array([1,1,2,3,4,5,6,6,6,6,0,2,2]), or 

    RETURNS:
        a list of tuples [(st, en),...,(st, en)] of same subarrays
    """
    segment_indices = []
    for k, g in groupby(enumerate(arr), key=lambda x: x[1]):
        segment = list(g)
        if len(segment) > 1:
            segment_indices.append((segment[0][0], segment[-1][0] + 1))
    return segment_indices


def remap_to_ordered_ids(ids_tensor):
    """ 
    maps cluster ids to temporally ordered ids for intelligible plotting
    """
    idslist = ids_tensor.clone().tolist()
    Maxid = max(np.unique(idslist))

    dic = {}
    j = 0
    for i in idslist:
        if i not in dic.keys():
            dic[i] = j
            j += 1
        else:
            pass
    return list(map(lambda x: abs(dic[x] - Maxid), idslist))


def get_labels(dat1, params, DATA, step_sz):
    behav = dat1 / 100 * params.K
    b = np.zeros_like(behav)
    b[(behav > 0.5) & (behav < 2.7)] = 1

    switched = False
    up = True

    for i in range(len(b)):
        if b[i] != 0:
            switched = False
            if up:
                b[i] = 1.0
            else:
                b[i] = -1.0

        if (b[i] == 0):
            if not switched:
                up = not up
                switched = True
    x = np.arange(0, len(DATA) * step_sz, step_sz)
    labels = b[0::step_sz][:len(x)]
    return x, labels


def saveForPPSeq(X_, cellid=0, path='datasets', name='ds'):
    with open(f'{path}/{name}_{cellid}.txt', 'w') as f:
        for i in range(X_.shape[0]):
            for j in range(X_.shape[1]):
                if X_[i, j] == 1:
                    f.write(f"{float(i+1)}\t{float(j)}\n")
