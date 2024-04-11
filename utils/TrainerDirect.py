from itertools import chain, groupby
from collections import Counter
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
from contextlib import nullcontext

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

from utils.nulldistDirect import NullDistEstimatorDirect
from utils.metrics import get_metrics, Metrics, MetaMetrics

from .utils import diff, tonp


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
        # self.evaluator = Metrics(params, GT, self.plotter.estimator.q_99)
        self.evaluator = MetaMetrics(self, params, GT)
        self.loss_dict = dict(
            ep=[],
            proj=[],
            proj_ent=[],
            TV=[],
            mu_ent=[],
            filt_xcor=[],
            proj_sim_cos=[],
            proj_xcor=[],
            filt_TV=[],
            loadings_loss=[],
            imbalance_loss=[],
        )
        self.EP = 0
        self.metrics = []
        self.proj_smoothing_ker = torch.ones(
            size=(
                1,
                1,
                1,
                self.params.proj_smoothing_kerwidth,
            ),
            device=params.device,
        ) / self.params.proj_smoothing_kerwidth
        # torch.save(model.state_dict(), '../checkpoints/untrained.torch')
        self.gaussian_target = self._get_gaussian_target()

    def _get_gaussian_target(self):
        x_values = torch.arange(self.params.W).to(self.params.device)
        sig = torch.tensor([16.0]).to(self.params.device)
        mu = torch.tensor([50.0]).to(self.params.device)
        return torch.exp(-torch.pow(x_values - mu, 2.) / (2 * torch.pow(sig, 2.)))

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

    def proj_xcorr__(self, a, b):
        a_ = a.view(1, 1, -1) - a.mean()
        b_ = b.view(1, 1, -1) - b.mean()
        l = len(a.squeeze())
        s0 = a_.std()
        s1 = b_.std()
        # return F.conv1d(a_, b_, padding=self.params.xcorr_padding).squeeze().var()
        return (F.conv1d(a_, b_, padding=self.params.xcorr_padding) / (s0 * s1) / l).pow(8).mean()
        # return (F.conv1d(a_.pow(6), b_.pow(6), padding=self.params.xcorr_padding) / (s0 * s1) / l).mean()

    def proj_xcorr(self, OUT):
        proj_xc = torch.tensor([0.0], device=OUT[0].device, dtype=OUT[0].dtype)
        c = 0
        for i, out0 in enumerate(OUT):
            for j, out1 in enumerate(OUT):
                if j > i:  # NOTE: only take the upper triangle, excluding the main diag
                    proj_xc = proj_xc + self.proj_xcorr__(out0, out1)  # NOTE: fixed accumulation
                    c += 1
        return proj_xc / c + torch.tensor([0.0020], device=proj_xc.device, dtype=proj_xc.dtype)  # NOTE: normalize

    def proj_cosine_similarity(self, OUT):
        c = 0
        sim = torch.tensor([0], device=self.params.device, dtype=torch.float)
        for i, out0_ in enumerate(OUT):
            for j, out1_ in enumerate(OUT):
                if j > i:  # NOTE: only take the upper triangle, excluding the main diag
                    out0 = F.conv2d(
                        out0_,
                        self.proj_smoothing_ker,
                        padding=(0, int(self.params.proj_smoothing_kerwidth / 2)),
                    )
                    out1 = F.conv2d(
                        out1_,
                        self.proj_smoothing_ker,
                        padding=(0, int(self.params.proj_smoothing_kerwidth / 2)),
                    )
                    sim += self.cossim(
                        out0.squeeze().flatten().pow(self.params.simpow),
                        out1.squeeze().flatten().pow(self.params.simpow),
                    )
                    c += 1
        return sim / c  # NOTE: normalize

    # FIXME: WIP
    def filt_xcorr(self, FILT):
        c = 0
        xcorr = torch.tensor([0], device=self.params.device, dtype=torch.float)
        for i in range(len(FILT)):
            for j in range(len(FILT)):
                if j > i:  # NOTE: only take
                    xcorr += F.conv2d(
                        FILT[i],  #.softmax(dim=-2),
                        FILT[j],  #.softmax(dim=-2),
                        padding=(0, self.params.W),
                    ).pow(2).mean()  #FIXME: which is better ? squeeze().var()
                    c += 1
        return xcorr / c  # NOTE: normalize

    def conv_power_imbalance(self, OUT):
        """ we take the convolutions, compute their power and treat these powers as a distribution.
            we want this distribution to be as far from peaky as possible. For this we maximize its
            entropy (i.e. miniize its negative entropy)
        """
        POW = torch.stack(OUT).squeeze().pow(2).mean(dim=1)
        entropy = -torch.sum(POW * POW.log())
        return -entropy

    def train(self, epochs=200, loadnull=False, freeze=None):

        pbar = trange(self.EP, epochs, desc=f'{0:.3f}\t{0:.3f}\t{0:.3f}\t{0:.3f}\t{0:.2f}', bar_format=bar_format)
        TTT = []
        for i in pbar:
            try:
                if self.params.enable_profiling:
                    profiler_context = torch.autograd.profiler.profile(
                        use_cuda=True if self.params.device.startswith('cuda') else False)
                else:
                    profiler_context = nullcontext()

                with profiler_context as prof:

                    tt = time.time()
                    self.optimizer.zero_grad()
                    # ttt = time.time()
                    OUT, FILT = self.model.forward(self.Xtorch)
                    # TTT.append(time.time() - ttt)

                    loss = torch.tensor([0], device=self.params.device, dtype=torch.float)

                    for k, (out, filt) in enumerate(zip(OUT, FILT)):
                        reconst = out.squeeze().var()
                        # reconst = out.sum() # wrong, out.pow(2).sum() works, but maximizes power

                        # NOTE: WIP
                        # entropy = Categorical(torch.softmax(
                        #     out.view(
                        #         self.params.out_channels,
                        #         -1,
                        #     ),
                        #     dim=1,
                        # )).entropy().sum()
                        entropy = torch.tensor([0.0], device=self.params.device, dtype=torch.float)

                        # NOTE: WIP
                        entropyOfMus = torch.tensor([0.0], device=self.params.device, dtype=torch.float)
                        # entropyOfMus = Categorical(
                        #     torch.softmax(
                        #         self.model.mus[k].view(
                        #             self.params.out_channels,
                        #             -1,
                        #         ),
                        #         dim=1,
                        #     )).entropy().sum()
                        # TV = diff(out.view(out_channels, -1), dim=1).pow(2).sum() # penalize the energy of the derivative ()
                        TV = diff(
                            out.view(
                                self.params.out_channels,
                                -1,
                            ),
                            dim=1,
                        ).pow(2).mean()  # penalize the energy of the derivative ()

                        filt_TV = filt.diff(dim=-1).pow(2).mean()  # filter TV

                        loss += -reconst  # maximize total filter response
                        loss += self.params.filt_TV_l * filt_TV
                        loss += self.params.TV_l * TV  # minimize proj total variation

                    # NOTE: WIP
                    if self.params.K == 1:
                        loadings_loss = torch.tensor([0.0], device=self.params.device, dtype=torch.float)
                    else:
                        loadings = list(self.model.parameters())[0]  # the the loadings tensor
                        loadings_loss = torch.tensor([0.0], device=self.params.device, dtype=torch.float)
                        for i, l0 in enumerate(loadings):
                            for j, l1 in enumerate(loadings):
                                if i > j:
                                    loadings_loss += torch.corrcoef(torch.vstack([l0, l1]).squeeze())[0, 1]

                    if (self.params.loadings_l > 0) and (self.params.K > 1):
                        loss += self.params.loadings_l * loadings_loss

                    # loss += self.params.proj_ent_l * entropy  # minimize proj entropy
                    # loss += -self.params.filt_ent_l * entropyOfMus  # maximize filt entropy

                    # add filter similarity loss

                    if len(OUT) > 1:
                        # sim_cos = self.JSD(OUT)  # NOTE: WIP

                        proj_sim_cos = self.proj_cosine_similarity(OUT)
                        proj_xcor = self.proj_xcorr(OUT)
                        filt_xcor = self.filt_xcorr(FILT)

                        loss += self.params.proj_sim_cos_l * proj_sim_cos  # minimize projection similarity
                        loss += self.params.proj_xcorr_l * proj_xcor  # minimize projection similarity
                        loss += self.params.filt_xcor_l * filt_xcor  # minimize filter similarity
                    else:
                        proj_sim_cos = torch.tensor([0])
                        proj_xcor = torch.tensor([0])
                        filt_xcor = torch.tensor([0])

                    # NOTE: WIP, adding imbalance loss
                    if ('imbalance_l' in self.params.keys()) and (self.params.K > 1):
                        imbalance_loss = self.params.imbalance_l * self.conv_power_imbalance(OUT)
                        loss += imbalance_loss
                    else:
                        imbalance_loss = torch.tensor([0.0], device=self.params.device, dtype=torch.float)

                    # for filt in FILT:
                    #     loss += 0.01 * F.mse_loss(
                    #         filt.squeeze().mean(dim=0),
                    #         self.gaussian_target,
                    #     )

                    # FIXME: return a dict of losses
                    self.loss_dict['ep'].append(self.EP)
                    self.loss_dict['proj'].append(reconst.item())
                    self.loss_dict['proj_ent'].append(entropy.item())
                    self.loss_dict['TV'].append(TV.item())
                    self.loss_dict['mu_ent'].append(entropyOfMus.item())
                    self.loss_dict['filt_xcor'].append(filt_xcor.item())
                    self.loss_dict['proj_sim_cos'].append(proj_sim_cos.item())
                    self.loss_dict['proj_xcor'].append(proj_xcor.item())
                    self.loss_dict['filt_TV'].append(filt_TV.item())
                    self.loss_dict['loadings_loss'].append(loadings_loss.item())
                    self.loss_dict['imbalance_loss'].append(imbalance_loss.item())

                    if freeze is not None:
                        self.model.filt.data = self.model.filt.data * freeze
                    loss.backward(retain_graph=False)
                    if freeze is not None:
                        self.model.filt.grad[freeze].zero_()

                    self.optimizer.step()

                    elapsed_s = time.time() - tt
                    if self.params.snapshot_interval is not None:
                        if i % self.params.snapshot_interval == 0:
                            proj = self.plotter.plot(i, self.loss_dict)
                    # mus.clamp_(min=1, max=99)

                    # metrics_dict = self.evaluator.get(tonp(OUT[0].squeeze()))
                    metrics_dict = self.evaluator.get(OUT)

                    self.metrics.append({
                        "ep": self.EP,
                        "TD": metrics_dict['true_positives'],
                        "FD": metrics_dict['false_positives'],
                        "ND": metrics_dict['false_negatives'],
                        "tpr": metrics_dict['tpr'],
                        "fpr": metrics_dict['fpr'],
                        "tnr": metrics_dict['tnr'],
                        "fnr": metrics_dict['fnr'],
                        "x": reconst.item() / self.params.Ts,
                    })

                    pbar.set_description(
                        f"proj: {reconst.item():.4f} | imb: {imbalance_loss.item():.4f} | TV: {TV.item():.4f} | fxc: {filt_xcor.item():.4f} | pcs: {proj_sim_cos.item():.4f}  | pxc: {proj_xcor.item():.4f} | load: {loadings_loss.item():.2f} | TP: {metrics_dict['true_positives']} | FP: {metrics_dict['false_positives']} | TN: {metrics_dict['true_negatives']} | FN: {metrics_dict['false_negatives']}"
                    )

                    # early stopping:
                    if metrics_dict['tpr'] >= self.params.early_stopping_pc:
                        cprint('Early termnination', color='yellow')
                        break

                    self.EP += 1
                # report profiling results
                if self.params.enable_profiling:
                    events = prof.key_averages()
                    try:
                        _name_ = [event.key for event in events]
                    except:
                        _name_ = None
                    try:
                        _cpu_time_total_ = [event.cpu_time_total for event in events]
                    except:
                        _cpu_time_total_ = None
                    try:
                        _cuda_time_total_ = [event.cuda_time_total for event in events]
                    except:
                        _cuda_time_total_ = None
                    try:
                        _input_shapes_ = [event.input_shapes for event in events]
                    except:
                        _input_shapes_ = None
                    try:
                        _cuda_memory_usage_ = [event.cuda_memory_usage for event in events]
                    except:
                        _cuda_memory_usage_ = None
                    try:
                        _flops_ = [event.flops for event in events]
                    except:
                        _flops_ = None

                    data_ = {
                        "name": _name_,
                        "cpu_total": _cpu_time_total_,
                        "cuda_total": _cuda_time_total_,
                        "input_shapes": _input_shapes_,
                        "cuda_memory_usage": _cuda_memory_usage_,
                        "flops": _flops_,
                        # Add more fields as needed
                    }

                    pd.DataFrame(data_).to_csv('AAprof_Direct.csv')

            except KeyboardInterrupt:
                cprint('User interrupt', color='green')
                break
        # cprint(np.mean(TTT), color='green')


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

        self.estimator = NullDistEstimatorDirect(X, params)
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
            # ax[1, 0].hist(self.model.mus[k].detach().cpu().numpy().squeeze(), bins=20)
            # ax[1, 1].hist(self.model.mus[k].detach().cpu().numpy().squeeze(), bins=20)
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

            # ax[0, 2].hist(self.estimator.nulldist_full, bins=50, color='blue', alpha=0.4)
            for q in (self.estimator.q_1, self.estimator.q_99):
                ax[0, 2].axvline(q)
            # ax[0, 2].twinx().hist(out, bins=50, color='red', alpha=0.4)

            # ax[0, 3].hist(self.estimator.nulldist, bins=50, color='blue', alpha=0.4)
            for q in np.quantile(self.estimator.nulldist, (0.001, 0.999)):
                ax[0, 3].axvline(q)
            ax[0, 3].axvline(out.sum(), color='red')
            # cprint(os.getcwd(), color='yellow')
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
                ax2.plot(loss_dict['proj'], label='Var(${X*C}$)')
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