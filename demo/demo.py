import torch, time, sys, os, argparse

sys.path.append('../')
from utils.constants import cm
import numpy as np
import torch.optim as optim
from utils.utils import load_config, tonp
from models.models import SeqNetDirect
from utils.TrainerDirect import Trainer

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams.update(load_config('../configs/rcparams.yaml'))

parser = argparse.ArgumentParser(description="demo")
parser.add_argument("--epochs", type=int, default=None)
args = parser.parse_args()


# load the parameters from YAML file
params = load_config('../configs/config_demo.yaml')
params.fs = 1 / params.dt  # sampling rate

# load dataset
dataset = np.load('../data/dataset_demo.npy', allow_pickle=True).item()
X_ = dataset['X_']
X_gt = dataset['X_gt']
GT = dataset['GT']
dat1 = dataset['dat1']

# update parameters with values calculated based on data size
end = X_.shape[1]
Ts = X_[:, :end].shape[1]
T = Ts * params.dt
t = np.arange(0, T, params.dt)
params.N = X_[:, :end].shape[0]
params.kernel_size = (params.N, params.W)
params.padding = tuple(params.padding)
params.Ts = Ts
params.N = X_.shape[0]
params.xlim = X_.shape[1]

# build the model and optimizer
model = SeqNetDirect(params).to(params.device)
optimizer = optim.Adam(model.parameters(), lr=0.1)  # tell the optimizer which var we want optimized

loadnull = False  # set False if you need to bootstrap the significance threshold
trainer = Trainer(model, optimizer, dat1, X_, GT, params, loadnull=loadnull)

Xtorch = torch.from_numpy(X_).view(1, 1, params.N, Ts).float().to(params.device)

# remember the null distribution
lim = 4000000
OUT, FILT = model.forward(Xtorch)
proj = tonp(OUT[0]).flatten()
nulldist = tonp(trainer.plotter.estimator.nulldist_full.flatten())[:lim]
dist_before = np.copy(proj)
proj_before = np.copy(proj)


# remove old images
for f in [fn for fn in os.listdir(params.path_to_data) if fn.endswith('.jpeg')]:
    os.remove(f'{params.path_to_data}/{f}')

# train the model
tt = time.time()
trainer.train(epochs=args.epochs, freeze=None)
print(f'{(time.time() - tt):.2f} sec.')

# get projection after the optimization
OUT, FILT = model.forward(Xtorch)
proj_after = tonp(OUT[0]).flatten()
dist_after = np.copy(proj_after)
proj_after = np.copy(proj_after)


########## Plot the results ##########

c0 = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0)
c1 = (1.0, 0.4980392156862745, 0.054901960784313725, 1.0)
fig, ax = plt.subplots(2,2, gridspec_kw=dict(width_ratios=[3,1]), sharey=True, sharex='col', figsize=(26*cm, 10*cm), dpi=300)

# before
ax[0,0].plot(proj_before, color='k')
ax[0,0].axhline(trainer.plotter.estimator.q_99, lw=2, ls=':')
ax[0,0].set_ylabel('$\hat{x}_t^{(k)}$, a.u.')


bins = np.linspace(0, 4, 100)
weights = np.ones_like(nulldist) / len(nulldist)

ax[0,1].hist(nulldist, bins=bins, weights=weights, orientation='horizontal', label='$p(\mathbf{x}_0^{(k)})$', histtype='step', lw=2)

weights = np.ones_like(dist_after) / len(dist_after)
ax[0,1].axhline(trainer.plotter.estimator.q_99, lw=2, ls=':')
handle1 = Line2D([], [], color=c0, lw=2) 
ax[0,1].legend(handles=[handle1], labels=['$p(\mathbf{x}_0^{(k)})$'], fontsize=14)


# after
ax[1,0].plot(proj_after, color='k')
ax[1,0].axhline(trainer.plotter.estimator.q_99, lw=2, ls=':')
ax[1,0].set_ylabel('$\hat{x}_t^{(k)}$, a.u.')


bins = np.linspace(0, 4, 100)
weights = np.ones_like(nulldist) / len(nulldist)

ax[1,1].hist(nulldist, bins=bins, weights=weights, orientation='horizontal', histtype='step', linewidth=2)

weights = np.ones_like(dist_after) / len(dist_after)
ax[1,1].hist(dist_after, bins=bins, weights=weights, orientation='horizontal', histtype='step', linewidth=2)
ax[1,1].axhline(trainer.plotter.estimator.q_99, lw=2, ls=':')
                       
    
ax[0,0].set_ylim(0, 3.5)
ax[0,1].set_ylim(0, 3.5)

ax[1,0].set_xlabel('Time step')
ax[1,1].set_xlabel('Density')


handle1 = Line2D([], [], color=c0, lw=2) 
ax[0,1].legend(handles=[handle1], labels=['$p(\mathbf{x}_0^{(k)})$'], fontsize=14)

handle1 = Line2D([], [], color=c0, lw=2) 
handle2 = Line2D([], [], color=c1, lw=2) # Match color and linewidth with hist2 
ax[1,1].legend(handles=[handle1, handle2], labels=['$p(\mathbf{x}_0^{(k)})$', '$p(\mathbf{x}^{(k)})$'], fontsize=14)

plt.savefig("../artifacts/Fig2_stats.pdf", bbox_inches='tight')
plt.savefig("../artifacts/Fig2_stats.png", bbox_inches='tight', dpi=300)

