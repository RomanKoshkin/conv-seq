import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from termcolor import cprint


def sm(input_tensor, temperature):
    scaled_input = input_tensor / temperature
    return F.softmax(scaled_input, dim=-1)


class SeqNetDirect(nn.Module):

    def __init__(self, params):
        super(SeqNetDirect, self).__init__()
        self.params = params
        self.loadings = nn.Parameter(torch.randn(size=(params.K, 1, params.N, 1), device=params.device))

        if self.params.with_clipping:
            # NOTE: with clipping
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=params.K,
                kernel_size=(params.N, params.W),
                bias=False,
                padding=self.params.padding,
                device=params.device,
            )
            self.EPS = 1e-10
            self.conv.weight.data.uniform_()
            # self.conv.weight.data.fill_(0.001)
            self.conv.weight.data.clamp_(min=self.EPS)
            self.conv.weight.data = self.conv.weight.data / (self.conv.weight.data.sum(dim=-1, keepdim=True) + self.EPS)

        else:
            # NOTE: with softmax
            self.filt = nn.Parameter(
                torch.randn(
                    size=(
                        self.params.K,
                        1,
                        self.params.N,
                        self.params.W,
                    ),
                    device=self.params.device,
                )
            )

    def forward(self, X):

        if self.params.with_clipping:
            # NOTE with clipping and normalization
            self.conv.weight.data.clamp_(min=self.EPS)
            self.conv.weight.data = self.conv.weight.data / (self.conv.weight.data.sum(dim=-1, keepdim=True) + self.EPS)
            # out = self.conv(X / X.sum(dim=-1, keepdim=True))  # NOTE: normalize by the number of spikes
            out = self.conv(X)  # NOTE: normalize by the number of spikes
            FILT = self.conv.weight.split(dim=0, split_size=1)
            OUT = out.split(dim=1, split_size=1)

        else:

            # NOTE with softmax (and maybe) loadings
            softmaxed_filt = self.filt.softmax(dim=-1)
            # softmaxed_filt = sm(self.filt, 3.0) # NOTE: temperature-scale softmax (WIP)
            if (self.params.loadings_l > 0) and (self.params.K > 1):
                softmaxed_filt = softmaxed_filt * self.loadings.sigmoid()
                # softmaxed_filt = softmaxed_filt * self.loadings.softmax(dim=-2)

            out = F.conv2d(
                X,
                weight=softmaxed_filt,  # NOTE: using the softmaxed filter
                padding=(0, self.params.padding),
                bias=None,
            )
            FILT = softmaxed_filt.split(dim=0, split_size=1)
            OUT = out.split(dim=1, split_size=1)

        return OUT, FILT
