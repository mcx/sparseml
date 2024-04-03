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


import torch

from sparseml.modifiers.quantization.utils import fake_quantize
from sparseml.pytorch.sparsification.quantization import QuantizationArgs


# Define the size of the tensor
rows = 1024
cols = 512


args = QuantizationArgs(
    num_bits=4,
    symmetric=True,
    group_size=1024,
    block_size=128,
    observer="foo",
    observer_kwargs={},
)


def test_fake_quantize():
    x = torch.rand(rows, cols)

    # get the values from Quantizer.find_params in gptq
    # or in our own Observer
    scale = 0
    zero_point = 0
    args = args

    fake_quantize(
        x,
        scale,
        zero_point,
        args,
    )
