import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
from .constants import bar_format
from termcolor import cprint
from .utils import tonp


class NullDistEstimatorDirect(object):

    def __init__(self, X, params):

        self.params = params
        cprint("NullDistEstimatorDirect", "yellow", "on_blue")

        self.Xtorch = torch.from_numpy(X).view(1, 1, params.N, params.Ts).float().to(params.device)
        self.Xtorch.requires_grad_(False)
        self.nulldist = []
        self.sample_means = torch.zeros(size=(self.params.niter, self.params.numsamples), device=self.params.device)
        self.sample_stds = torch.zeros(size=(self.params.niter, self.params.numsamples), device=self.params.device)
        self.nulldist_full = torch.zeros(
            size=(self.params.niter, self.params.numsamples, self.params.Ts), device=self.params.device
        )
        self.EPS = 1e-10

    def sample(self):
        with torch.no_grad():
            self.filt = torch.randn(
                size=(
                    self.params.numsamples,
                    1,
                    self.params.N,
                    self.params.W,
                ),
                device=self.params.device,
            )

            if self.params.with_clipping:
                # NOTE: with clamping
                self.filt.clamp_(min=self.EPS)
                self.filt = self.filt / (self.filt.sum(dim=-1, keepdim=True) + self.EPS)

            else:
                # NOTE: with softmax
                # self.filt = self.filt.softmax(dim=-1)
                softmaxed_filt = self.filt.softmax(dim=-1)

                loadings = torch.randn(
                    size=(
                        self.params.numsamples,
                        1,
                        self.params.N,
                        1,
                    ),
                    device=self.params.device,
                )

                if (self.params.loadings_l > 0) and (self.params.K > 1):
                    softmaxed_filt = softmaxed_filt * loadings.sigmoid()
                else:
                    pass

            out = F.conv2d(
                self.Xtorch,  # / self.Xtorch.sum(dim=-1, keepdim=True),  # NOTE: normalize by the number of spikes
                weight=softmaxed_filt,
                padding=(0, self.params.padding),
                bias=None,
            ).squeeze()

            summedout = out.sum(dim=1)
            sample_mean = out.mean(dim=-1).flatten()
            sample_std = out.std(dim=-1, correction=1).flatten()

        return summedout, sample_mean, sample_std, out[:, : self.params.Ts]

    def estimate(self, loadnull=False):
        if not loadnull:
            for i in trange(self.params.niter, desc="Estimating null distribution | ", bar_format=bar_format):
                summedout, sample_mean, sample_std, out = self.sample()
                # print(summedout.shape, sample_mean.shape, sample_std.shape, out.shape)
                self.nulldist += summedout
                self.sample_means[i, :] = sample_mean
                self.sample_stds[i, :] = sample_std
                try:
                    self.nulldist_full[i, :, :] = out
                except Exception as e:
                    cprint("Did you set the params.padding right?", color="red", attrs=["bold"])
                    raise e

            self.nulldist = torch.stack(self.nulldist).flatten().detach().cpu().numpy()
            sampling_mean = self.sample_means.mean().item()
            sampling_std = self.sample_stds.mean().item()
            self.q_1, self.q_99 = 0, sampling_mean + self.params.significance_sigmas * sampling_std
            # torch.save({
            #     "null_dist": self.nulldist,
            #     "q_1": self.q_1,
            #     "q_99": self.q_99
            # }, "../checkpoints/nulldist.torch")
        else:
            print("loading nulldist")
            d = torch.load("../checkpoints/nulldist.torch")
            self.nulldist = d["null_dist"]
            self.q_1 = d["q_1"]
            self.q_99 = d["q_99"]
