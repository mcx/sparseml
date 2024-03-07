# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

from sparseml.modifiers.utils.compression_wrapper import ModuleCompressionWrapper


try:
    import transformers
except ImportError as err:
    transformers = None
    transformers_err = err

import logging
import math

import torch
import torch.nn as nn


__all__ = ["IncohSparseGptWrapper"]

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# Incoherence Processing
# =============================================================================

def FFT(X, phase, inverse=False):
    "operates on 2nd dimension"
    n = X.shape[-1]
    assert n % 2 == 0
    assert len(phase) == n//2
    output = X.float().view(-1, n//2, 2)
    output = torch.view_as_complex(output.contiguous())
    if not inverse:
        output = output * phase
        output = torch.fft.fft(output, dim=-1, norm='ortho')
    else:
        output = torch.fft.ifft(output, dim=-1, norm='ortho')
        output = output * torch.conj(phase)
    output = torch.view_as_real(output)
    return output.view(-1, n)


def FFT_H(H, SU):
    return FFT(FFT(H, phase=SU).T, phase=SU).T


def FFT_W(W, SU, SV):
    return FFT(FFT(W.T, phase=SV).T, phase=SU)


def FFT_W_inv(W, SU, SV):
    return FFT(
        FFT(W, SU, inverse=True).T,
        SV, inverse=True).T

# =============================================================================
# =============================================================================



class IncohSparseGptWrapper(ModuleCompressionWrapper):
    """
    Runs SparseGPT on a single module that contains no sub-modules

    Lifecycle:
        - add_batch
        - fasterprune
        - free

    :param name: name of module to run compression on
    :param layer: module to run compression on
    """

    def __init__(self, name, layer):
        super().__init__(name=name, layer=layer)

        # for Hessian calculation
        self.register_buffer(
            "H", torch.zeros((self.columns, self.columns), device=self.dev)
        )

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        Add a batch of layer input and output data to the Hessian calculation

        :param inp: tensor containing layer input
        :param out: tensor containing layer output
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t()).to(self.dev)

    def fasterprune(
        self,
        sparsity: float,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ):
        """
        Run pruning and quantization(if applicable) on the layer up to the target
        sparsity value.

        :param sparsity: target sparsity to reach for layer
        :param prunen: N for N:M pruning
        :param prunem: M for N:M pruning
        :param blocksize: Number of columns to compress in one pass
        :param percdamp: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
        """
        final_shape = self.layer.weight.shape
        final_dtype = self.layer.weight.dtype
        W = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        dead = torch.diag(self.H) == 0
        self.H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(self.H))
        diag = torch.arange(self.columns, device=self.dev)
        self.H[diag, diag] += damp

        # =================================================
        # Incoherence Processing
        # =================================================
        import copy
        Worig = copy.deepcopy(W).cpu()
        Horig = copy.deepcopy(self.H).cpu()

        # check incoherence 
        if True:
            Wij_max = W.abs().max().item()
            _, eigH = torch.linalg.eigh(self.H)
            Qij_max = eigH.abs().max().item()

        (m, n) = W.shape
        theta_SU = 2 * math.pi * torch.rand(n//2).to(self.dev)
        theta_SV = 2 * math.pi * torch.rand(m//2).to(self.dev)
        SU = torch.complex(theta_SU.cos(), theta_SU.sin())
        SV = torch.complex(theta_SV.cos(), theta_SV.sin())
        self.H = FFT_H(self.H, SU)
        W = FFT_W(W, SU, SV)
        SU = SU.cpu()
        SV = SV.cpu()

        if True:
            hatWij_max = W.abs().max().item()
            _, hateigH = torch.linalg.eigh(self.H)
            hatQij_max = hateigH.abs().max().item()
            _LOGGER.info(f"max|W_ij|: {Wij_max:.3f} / {hatWij_max:.3f}")
            _LOGGER.info(f"max|Q_ij|: {Qij_max:.3f} / {hatQij_max:.3f}")
        # =================================================
        # =================================================


        self.H = torch.linalg.cholesky(self.H)
        self.H = torch.cholesky_inverse(self.H)
        self.H = torch.linalg.cholesky(self.H, upper=True)
        Hinv = self.H

        mask = None

        # See section 3.4 of https://arxiv.org/abs/2203.07259
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = (
                        W1[:, i : (i + prunem)] ** 2
                        / (torch.diag(Hinv1)[i : (i + prunem)].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self.layer, "weight_fake_quant"):
                    scale = self.layer.weight_fake_quant.scale
                    zero_point = self.layer.weight_fake_quant.zero_point
                    dtype = self.layer.weight_fake_quant.dtype
                    qscheme = self.layer.weight_fake_quant.qscheme
                    if qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                        q = torch.quantize_per_tensor(q, scale, zero_point, dtype)
                    else:
                        q = torch.quantize_per_channel(q, scale, zero_point, 0, dtype)
                    q = torch.dequantize(q)

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        _LOGGER.info("time %.2f" % (time.time() - tick))
        _LOGGER.info("error %.2f" % torch.sum(Losses).item())

        # =======================================
        # =======================================
        # Incoherence Processing
        W = FFT_W_inv(W, SU.to(self.dev), SV.to(self.dev))

        # Comparing proxy loss 
        Worig = Worig.to(self.dev)
        Horig = Horig.to(self.dev)
        err_frob = (W - Worig).square().sum() / Worig.square().sum()
        err_proxy = (((W - Worig) @ Horig) * (W - Worig)).sum() / ((Worig @ Horig) * Worig).sum()
        _LOGGER.info(f"frob  error: {err_frob:.5f}")
        _LOGGER.info(f"proxy error: {err_proxy:.5f}")
        # =====================================================================
        # =====================================================================

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.reshape(final_shape).to(final_dtype)

        # This is a bit hacky, but FSDP updates only work if we change the weight in
        # place, clone() or direct assignment won't work
        self.layer.weight -= self.layer.weight
        self.layer.weight += W

    def free(self):
        """
        Free the Hessian memory after the layer is complete
        """
        delattr(self, "H")
        super().free()
