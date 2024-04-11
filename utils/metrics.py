from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from termcolor import cprint


def tonp(x):
    return x.cpu().detach().numpy()


class Metrics:
    #     trainer.plotter.estimator.q_99
    def __init__(self, params, GT, alpha):
        self.alpha = alpha
        self.params = params

        A = GT[0]
        self.tp_zones, self.fp_zones = [], []

        tp_half_width = params.W

        for mid in A:
            self.tp_zones.append((int(mid - tp_half_width), int(mid + tp_half_width)))

        for i in range(len(self.tp_zones)):
            if i == 0:
                self.fp_zones.append((0, self.tp_zones[i][0]))
            elif i == len(self.tp_zones) - 1:
                self.fp_zones.append((self.tp_zones[i][1], params.Ts - 1))
            else:
                self.fp_zones.append((self.tp_zones[i - 1][1], self.tp_zones[i][0]))

    def get(self, proj):
        peaks, _ = find_peaks(proj, height=self.alpha, distance=self.params.W)

        TPZ = np.zeros((len(self.tp_zones),))
        FPZ = np.zeros((len(self.fp_zones),))

        for p in peaks:
            for i, (st, en) in enumerate(self.tp_zones):
                if (st <= p) and (p < en):
                    TPZ[i] = 1.0
            for i, (st, en) in enumerate(self.fp_zones):
                if (st <= p) and (p < en):
                    FPZ[i] = 1.0

        P = len(TPZ)  # condition positive
        N = len(FPZ)  # condition negative

        true_positives = sum(TPZ == 1)
        false_positives = sum(FPZ == 1)
        false_negatives = sum(TPZ == 0)
        true_negatives = sum(FPZ == 0)

        tpr = true_positives / P
        fpr = false_positives / N
        fnr = false_negatives / P
        tnr = true_negatives / N

        assert tpr + fnr == 1, 'check metrics.get'
        assert fpr + tnr == 1, 'check metrics.get'

        metrics_dict = dict(
            tpr=tpr,
            fpr=fpr,
            tnr=tnr,
            fnr=fnr,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
        )
        return metrics_dict


class NewMetrics:

    def __init__(self, params, gt, alpha):
        self.alpha = alpha
        self.params = params

        A = gt
        self.tp_zones, self.fp_zones = [], []

        tp_half_width = params.W

        for mid in A:
            self.tp_zones.append((int(mid - tp_half_width), int(mid + tp_half_width)))

        for i in range(len(self.tp_zones)):
            if i == 0:
                self.fp_zones.append((0, self.tp_zones[i][0]))
            elif i == len(self.tp_zones) - 1:
                self.fp_zones.append((self.tp_zones[i][1], params.Ts - 1))
            else:
                self.fp_zones.append((self.tp_zones[i - 1][1], self.tp_zones[i][0]))

    def get(self, proj):
        peaks, _ = find_peaks(proj, height=self.alpha, distance=self.params.W)

        TPZ = np.zeros((len(self.tp_zones),))
        FPZ = np.zeros((len(self.fp_zones),))

        for p in peaks:
            for i, (st, en) in enumerate(self.tp_zones):
                if (st <= p) and (p < en):
                    TPZ[i] = 1.0
            for i, (st, en) in enumerate(self.fp_zones):
                if (st <= p) and (p < en):
                    FPZ[i] = 1.0

        P = len(TPZ)  # condition positive
        N = len(FPZ)  # condition negative

        true_positives = sum(TPZ == 1)
        false_positives = sum(FPZ == 1)
        false_negatives = sum(TPZ == 0)
        true_negatives = sum(FPZ == 0)

        tpr = true_positives / P
        fpr = false_positives / N
        fnr = false_negatives / P
        tnr = true_negatives / N

        assert tpr + fnr == 1, 'check metrics.get'
        assert fpr + tnr == 1, 'check metrics.get'

        metrics_dict = dict(
            tpr=tpr,
            fpr=fpr,
            tnr=tnr,
            fnr=fnr,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
        )
        return metrics_dict


def get_map(params, GT, OUT, alpha):
    Z = np.zeros((params.K, len(GT)))

    for filt_ID in range(params.K):
        for pattern_ID in range(len(GT)):
            proj = tonp(OUT[filt_ID]).squeeze()
            peaks, _ = find_peaks(proj, height=alpha, distance=params.W // 2)
            MINDISTS = []
            for p in peaks:
                d = np.min(np.abs(np.array(GT[pattern_ID]) - p))
                MINDISTS.append(d)
            Z[filt_ID, pattern_ID] = np.mean(MINDISTS) if len(peaks) > 10 else 1000

    a = np.array(list(zip(Z.argmin(axis=1), Z.min(axis=1))))
    b = pd.DataFrame(a).sort_values(by=[0, 1], ascending=True).dropna()
    c = b.drop_duplicates(subset=b.columns[0], keep='first')
    return {k: pattern_id for k, pattern_id in zip(c.index.to_numpy(), c[0].to_numpy().astype(int))}


class MetaMetrics:

    def __init__(self, trainer, params, GT):
        self.GT = GT
        self.params = params
        self.alpha = trainer.plotter.estimator.q_99
        self.METRICS = {pattern_ID: NewMetrics(params, GT[pattern_ID], self.alpha) for pattern_ID in range(len(GT))}

    def get(self, OUT):
        out = dict(
            tpr=0,
            fpr=0,
            tnr=0,
            fnr=0,
            true_positives=0,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
        )
        TPR, FPR, TNR, FNR = [], [], [], []

        mapdict = get_map(self.params, self.GT, OUT, self.alpha)

        for filt_ID, pattern_ID in mapdict.items():
            metrics = self.METRICS[pattern_ID].get(tonp(OUT[filt_ID]).squeeze())
            TPR.append(metrics['tpr'])
            FPR.append(metrics['fpr'])
            TNR.append(metrics['tnr'])
            FNR.append(metrics['fnr'])
            out['tpr'] = np.mean(TPR)
            out['fpr'] = np.mean(FPR)
            out['tnr'] = np.mean(TNR)
            out['fnr'] = np.mean(FNR)
            out['true_positives'] += metrics['true_positives']
            out['false_positives'] += metrics['false_positives']
            out['true_negatives'] += metrics['true_negatives']
            out['false_negatives'] += metrics['false_negatives']

        return out


def get_metrics(out, A, q_99, params, debug=False, X_=None, xlim=(0, 4000)):
    """ 
    LEGACY
    """

    peaks, _ = find_peaks(out, height=q_99, distance=params.W)

    I = []
    for i, p in enumerate(peaks):
        if out[p] > q_99:  # if the height of that peak is high
            I.append(i)
    peaks = peaks[I]

    TP, FP, FN = 0, 0, 0

    if debug:
        fig, ax = plt.subplots(1, 1, figsize=(18, 4))
        axt = ax.twinx()
        ax.spy(X_, aspect='auto', origin='lower', markersize=0.7, color='k')
        axt.plot(out)
        axt.axhline(q_99, lw=3, ls=':', color='red')
        axt.plot(peaks, out[peaks], 'r*', markersize=30)
        for mid in A:
            axt.axvline(mid, lw=1, color='red')
        ax.set_xlim(xlim)
        axt.set_ylim(0, 3 * q_99)

    for p in peaks:  # p are peak indexes
        distToClosestRealSeq = min(abs(p - np.array(A)))  # distance to closess seq_mid from current peak
        if distToClosestRealSeq < params.W:  # if dist is small enough, discovery
            TP += 1
        else:
            FP += 1

    if not len(peaks) == 0:
        for mid in A:  # mid are seq_mids
            distToClosestRealSeq = min(abs(mid - peaks))  # distance to closess peak from current seq_mid
            closestPeakIdx = np.argmin(abs(mid - peaks))
            if distToClosestRealSeq > params.W:  # if that distance is large, non-discovery
                FN += 1
                if debug:
                    plt.axvline(peaks[closestPeakIdx], lw=2, color='green')
            else:  # else check if the peak is significant
                if out[peaks[closestPeakIdx]] < q_99:
                    FN += 1
                    if debug:
                        plt.axvline(peaks[closestPeakIdx], lw=2, color='green')
    else:
        return 0, 0, len(A)  # if no significan peaks exist, return all non-discovered

    return TP, FP, FN


class PPSeqMetrics:

    def __init__(self, GT, Ts, W):

        A = GT[0]
        self.tp_zones, self.fp_zones = [], []

        tp_half_width = W

        for mid in A:
            self.tp_zones.append((int(mid - tp_half_width), int(mid + tp_half_width)))

        for i in range(len(self.tp_zones)):
            if i == 0:
                self.fp_zones.append((0, self.tp_zones[i][0]))
            elif i == len(self.tp_zones) - 1:
                self.fp_zones.append((self.tp_zones[i][1], Ts - 1))
            else:
                self.fp_zones.append((self.tp_zones[i - 1][1], self.tp_zones[i][0]))

    def get(self, peaks):

        TPZ = np.zeros((len(self.tp_zones),))
        FPZ = np.zeros((len(self.fp_zones),))

        for p in peaks:
            for i, (st, en) in enumerate(self.tp_zones):
                if (st <= p) and (p < en):
                    TPZ[i] = 1.0
            for i, (st, en) in enumerate(self.fp_zones):
                if (st <= p) and (p < en):
                    FPZ[i] = 1.0

        P = len(TPZ)  # condition positive
        N = len(FPZ)  # condition negative

        true_positives = sum(TPZ == 1)
        false_positives = sum(FPZ == 1)
        false_negatives = sum(TPZ == 0)
        true_negatives = sum(FPZ == 0)

        tpr = true_positives / P
        fpr = false_positives / N
        fnr = false_negatives / P
        tnr = true_negatives / N

        assert tpr + fnr == 1, 'check metrics.get'
        assert fpr + tnr == 1, 'check metrics.get'

        metrics_dict = dict(
            tpr=tpr,
            fpr=fpr,
            tnr=tnr,
            fnr=fnr,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
        )
        return metrics_dict