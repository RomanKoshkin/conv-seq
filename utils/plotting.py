from email.utils import collapse_rfc2231_value
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from .utils import remap_to_ordered_ids, load_config, tonp

# matplotlib.rcParams.update(load_config('../configs/rcparams.yaml'))

import torch
from dataclasses import dataclass, field


@dataclass
class Figure:

    def __init__(self) -> None:
        prefix: field(default_factory=str)
        X: field(default_factory=np.ndarray)
        Z: field(default_factory=np.ndarray)
        ids: field(default_factory=torch.tensor)
        step_sz: field(default_factory=int)
        logits: field(default_factory=torch.tensor)


def save_data_for_figure(prefix, X, Z, ids, step_sz, params):
    a = Figure()
    a.prefix = prefix
    a.X = X
    a.Z = Z
    a.ids = ids
    a.step_sz = step_sz
    a.logits = torch.rand(size=(len(ids), params.K))
    torch.save(a, f'../data/Figure_data_for_{prefix}.dat')
    print('saved')


def plot_raster(X, ids, step_sz, logits, params, LIM, ax):
    ax.spy(X, aspect='auto', markersize=0.5, origin='lower', color='black')
    axt = ax.twinx()
    axt.plot(
        np.arange(0,
                  len(ids.detach().cpu().numpy()) * step_sz, step_sz),
        remap_to_ordered_ids(ids),
        # tonp(ids),
        lw=1.5,
        label='K-means cluster IDs')
    preds = remap_to_ordered_ids(logits.argmax(dim=1))
    # axt.plot(np.arange(0, len(ids) * step_sz, step_sz), preds, 'bo', alpha=0.3, label='argmax(logits)')

    # axt.plot(dat1/100*params.K, 'r')

    # axt.plot(x, labels.detach().cpu().numpy(), 'go', alpha=0.2, label='ideals')
    ax.set_xlim(LIM)
    ax.set_ylabel('Neuron IDs')

    axt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.9))
    axt.set_ylabel('Kmeans cluster IDs')
    # ax.axvspan(2900, 2900 + params.W, alpha=0.2)

    # tmp = logits.argmax(dim=1).diff(dim=0)
    # tmp[tmp != 0] = 1
    # ax.plot(np.arange(0, len(tmp)*step_sz, step_sz), tmp.tolist(), alpha=0.2)

    # axt.plot(np.arange(0, len(ids)*step_sz, step_sz), embeds[:, 0].detach().cpu(), lw=1, alpha=0.5)
    # axt.plot(np.arange(0, len(ids)*step_sz, step_sz), embeds[:, 1].detach().cpu(), lw=1, alpha=0.5)
    return axt


def Plot(embeds_np, X, ids, step_sz, logits, params, LIM):
    # create figure and gridspec
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(nrows=2, ncols=4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 2], wspace=0.1, hspace=0.3)

    # add subplots
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, :])

    ax0.scatter(embeds_np[:, 0], embeds_np[:, 1], c=ids.detach().cpu().numpy())
    ax0.set_xlim(0, 2)
    ax0.set_ylim(0, 2)
    ax1.scatter(embeds_np[:, 0], embeds_np[:, 1], c=ids.detach().cpu().numpy())
    ax2.scatter(embeds_np[:, 1], embeds_np[:, 2], c=ids.detach().cpu().numpy())
    ax3.scatter(embeds_np[:, 0], embeds_np[:, 2], c=ids.detach().cpu().numpy())
    axt = plot_raster(X, ids, step_sz, logits, params, LIM, ax4)
    return axt
