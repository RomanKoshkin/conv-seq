import matplotlib
import matplotlib.pyplot as plt
import sys, argparse

sys.path.append('../')
import numpy as np
import scipy.io as sio
from utils.utils import load_config
from utils.synthetic import get_background


parser = argparse.ArgumentParser(description="make_dataset")
parser.add_argument("--p_drop", type=float, default=None)
parser.add_argument("--gap_ts", type=int, default=None)
parser.add_argument("--seqlen", type=int, default=None)
parser.add_argument("--jitter_std", type=float, default=None)
args = parser.parse_args()

matplotlib.rcParams.update(load_config('../configs/rcparams.yaml'))


# load the parameters from YAML file
params = load_config('../configs/config_demo.yaml')
params.fs = 1 / params.dt  # sampling rate

# load data
dat1 = sio.loadmat(f'{params.path_to_data}/position_per_frame.mat')['position_per_frame'].flatten().astype('float')
dat1 = dat1 / dat1.max() * 96  # make sure the position ranges from 0 to 96 cm (as in the paper)
X = sio.loadmat(f'{params.path_to_data}/neuronal_activity_mat.mat')['neuronal_activity_mat']


params.xlim = X.shape[1]

# make background activity
X_ = get_background(np.copy(X))


# embed sequences at set intervals, of set duration on set neurons:
p_drop = args.p_drop # 0.1
gap_ts = args.gap_ts # 400
seqlen = args.seqlen # 120
jitter_std = args.jitter_std # 10


second_sequence_starts_at_neuron = seqlen + 50
second_sequence_starts_at_t = seqlen + 280


X_ = get_background(np.copy(X))

####################################################################################
end = X_.shape[1]
Ts = X_[:, :end].shape[1]

seqA_gt = []
seqB_gt = []
seqC_gt = []
for st in range(0, Ts-gap_ts, gap_ts):
    try:
        for i, disp in enumerate(range(st, st+seqlen)):
            jitter = np.round(np.random.randn() * jitter_std).astype(int)
            j = disp+jitter
            if j < Ts:
                pass
                
                
                choice = np.random.choice([0, 1], p=[p_drop, 1-p_drop])
                if (choice == 1) and (X_[:, j][seqlen:].sum() > 0):
                    to_be_dropped = np.where(X_[:, j])[0]
                    to_be_dropped = to_be_dropped[to_be_dropped >= seqlen]
                    X_[i, j] = 1
                    X_[to_be_dropped[0], j] = 0 # drop a spike at this time to maintain the original f_rate
        seqA_gt.append(np.round(np.mean([st, j])))
    except Exception as e:
        print(e)
    
    
GT = [seqA_gt]

colors = ['red', 'green', 'cyan']
fig, ax = plt.subplots(3, 1, figsize=(18,6), sharex=True)
ax[0].spy(X, aspect='auto', markersize=1.5, origin='lower')
ax[2].plot(X.mean(axis=0), alpha=0.4)
ax[2].plot(X_.mean(axis=0), alpha=0.4)


perm = np.arange(X_.shape[0])
np.random.shuffle(perm)

X_gt = np.copy(X_)
X_ = X_[perm, :]
ax[1].spy(X_, aspect='auto', markersize=1.5, origin='lower')

for i in seqA_gt:
    ax[1].axvline(i, color='red')
    
# for i in seqB_gt:
#     ax[1].axvline(i, color='green')
    
ax[0].set_xlim(0, 5000)
print(len(seqA_gt))


np.save(f'{params.path_to_data}/dataset_demo.npy',
        {
            'X_': X_,
            'X_gt': X_gt,
            'GT': GT,
            'dat1': dat1,
        }
)
