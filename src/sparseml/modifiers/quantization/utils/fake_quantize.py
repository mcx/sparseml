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

from typing import Dict

import torch
from pydantic import BaseModel


class QuantizationArgs(BaseModel):
    n_bits: int
    group_size: int  # -1 means channel wise
    block_size: int
    symmetric: bool
    observer: str
    observer_kwargs: Dict


# ref: https://pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html
def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
) -> torch.Tensor:
    max_q = torch.tensor(2**args.n_bits - 1)
    columns = x.shape[1]
    for i1 in range(0, columns, args.block_size):
        i2 = min(i1 + args.block_size, columns)
        count = i2 - i1

        W1 = x[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)

        for i in range(count):
            w = W1[:, i]

            if args.group_size != -1:
                if (i1 + i) % args.group_size == 0:
                    xmin, xmax = get_qparams(
                        x[:, (i1 + i) : (i1 + i + args.group_size)], args.symmetric
                    )
                    scale, zero = get_scale_zero_point(
                        x[:, (i1 + i) : (i1 + i + args.group_size)],
                        max_q,
                        xmax,
                        xmin,
                        args.symmetric,
                        args.group_size,
                    )

            q = quantize(w.unsqueeze(1), scale, zero, max_q).flatten()
            Q1[:, i] = q

    x_q = quantize(x, scale, zero_point, max_q)
    return dequantize(x_q, scale, zero_point)


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_max: torch.Tensor,
) -> torch.Tensor:
    return torch.clamp(
        round(
            x / scale + zero_point,
            0,
            q_max,
        )
    )


def dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    return (x_q - zero_point) * scale


# move to observer
def get_scale_zero_point(
    x: torch.Tensor,
    max_q: int,
    xmax: torch.Tensor,
    xmin: torch.Tensor,
    is_symmetric: bool,
    group_size: int,
):
    if max_q < 0:
        scale = xmax
        zero = xmin
    else:
        scale = (xmax - xmin) / max_q
        if is_symmetric:
            zero = torch.full_like(scale, (max_q + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)

    # channel-wise quant
    if group_size == -1:
        tmp = x.shape[0]
        scale = scale.repeat(tmp)
        zero = zero.repeat(tmp)

    shape = [-1] + [1] * (len(shape) - 1)
    scale = scale.reshape(shape)
    zero = zero.reshape(shape)

    return scale, zero


# move to observer
def get_qparams(x: torch.Tensor, is_symmetric: bool):
    tmp = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    if is_symmetric:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmin < 0
        if torch.any(tmp):
            xmin[tmp] = -xmax[tmp]
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    return xmin, xmax
