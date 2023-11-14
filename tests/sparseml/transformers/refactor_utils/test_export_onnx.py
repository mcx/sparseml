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

import onnx
import pytest
import torch

from src.sparseml.transformers.refactor_utils.export_onnx import (
    export_module_to_deployment_target,
    validate_input_optimizations,
)
from tests.sparseml.pytorch.helpers import ConvNet


@pytest.fixture()
def model():
    return ConvNet()


@pytest.fixture()
def sample_batch():
    return torch.randn(1, 3, 28, 28)


@pytest.mark.parametrize(
    "deployment_target", ["onnx", "tensorrt", "deepsparse", "unknown"]
)
def test_export_module_to_deployment_target(model, deployment_target, sample_batch):
    if deployment_target == "unknown":
        with pytest.raises(ValueError):
            export_module_to_deployment_target(model, deployment_target, sample_batch)
    elif deployment_target == "tensorrt":
        with pytest.raises(NotImplementedError):
            export_module_to_deployment_target(model, deployment_target, sample_batch)
    else:
        model = export_module_to_deployment_target(
            model, deployment_target, sample_batch
        )
        assert isinstance(model, onnx.ModelProto)


@pytest.mark.parametrize("optimization_name", ["kv_cache_injection", "unknown"])
def test_validate_input_optimizations(optimization_name):
    validate_input_optimizations(optimization_name)
