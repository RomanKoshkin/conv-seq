import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np


class NullDistEstimator(object):

    def __init__(self, X, params):

        self.params = params
        # self.sigmas = torch.tensor([params.sigma], dtype=torch.float, requires_grad=False,
        #                            device=params.device).repeat(params.N)  # repeat is almost free
        self.sigma = torch.tensor([params.sigma], dtype=torch.float, requires_grad=False, device=params.device)
        self.x_values = torch.linspace(0, params.W, params.W, requires_grad=False, device=params.device).float()
        self.Xtorch = torch.from_numpy(X).view(1, 1, params.N, params.Ts).float().to(params.device)
        self.Xtorch.requires_grad_(False)
        self.filt = torch.zeros(
            size=(params.numsamples, 1, params.N, params.W),
            requires_grad=False,
            device=params.device,
        )
        self.nulldist = []
        self.nulldist_full = []

    # def gaussian_proto(self, mu, sig):
    #     return torch.exp(-torch.pow(self.x_values - mu, 2.) / (2 * torch.pow(sig, 2.)))

    def gaussian(self, mus):
        return torch.exp(
            -torch.pow(self.x_values.repeat(self.params.N, 1) - mus.unsqueeze(-1), 2.0)
            / (2 * torch.pow(self.sigma, 2.0))
        )

    def sample(self):
        for k in range(self.params.numsamples):
            mus = torch.randint(
                self.params.W,
                size=(self.params.N,),
                device=self.params.device,
                requires_grad=False,
            ).float()
            # for i, (mu, sigma) in enumerate(zip(mus, self.sigmas)):
            #     self.filt[k, 0, i, :] = self.gaussian(mu, sigma)
            self.filt[k, 0, :, :] = self.gaussian(mus)

        # print(self.filt.shape)
        # raise
        out = F.conv2d(
            self.Xtorch,
            self.filt,
            padding=(0, self.params.padding),
        ).squeeze()
        summedout = out.sum(dim=1)
        fullout = out.max(dim=1)[0]

        return summedout, fullout

    def estimate(self, loadnull=False):
        if not loadnull:
            for i in trange(self.params.niter, desc="Estimating null distribution | "):
                summedout, fullout = self.sample()
                self.nulldist += summedout
                self.nulldist_full += fullout
            self.nulldist = torch.stack(self.nulldist).flatten().detach().cpu().numpy()
            self.nulldist_full = torch.stack(self.nulldist_full).detach().cpu().numpy()
            self.q_1, self.q_99 = np.quantile(self.nulldist_full, (0.001, self.params.q99))
            torch.save(
                {"null_dist": self.nulldist, "nulldist_full": self.nulldist_full, "q_1": self.q_1, "q_99": self.q_99},
                "../checkpoints/nulldist.torch",
            )
        else:
            print("loading nulldist")
            d = torch.load("../checkpoints/nulldist.torch")
            self.nulldist = d["null_dist"]
            self.nulldist_full = d["nulldist_full"]
            self.q_1 = d["q_1"]
            self.q_99 = d["q_99"]
