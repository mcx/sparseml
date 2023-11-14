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

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import onnx
from torch.nn import Module

from sparseml.exporters.kv_cache_injector import KeyValueCacheInjector
from sparseml.exporters.onnx_to_deepsparse import ONNXToDeepsparse
from sparseml.pytorch.opset import TORCH_DEFAULT_ONNX_OPSET
from sparseml.pytorch.torch_to_onnx_exporter import TorchToONNX


AVAILABLE_OPTIMIZATIONS = {
    "kv_cache_injection",
    "convert_qat",
    "disable_bn_fusing",
    "skip_input_quantize",
}
_LOGGER = logging.getLogger(__name__)

# TODO: Stress test to make sure that performance is okay
def export_onnx(
    model: Module,
    sample_batch: Any,
    file_path: str,
    deployment_target: str = "deepsparse",
    opset: int = TORCH_DEFAULT_ONNX_OPSET,
    optimizations: Optional[List[str]] = None,
):
    validate_input_optimizations(optimizations)
    model = export_module_to_deployment_target(
        model, deployment_target, optimizations, opset, sample_batch
    )
    if "kv_cache_injection" in optimizations:
        model = apply_kv_cache_injection(model)


def export_module_to_deployment_target(
    model: Module,
    deployment_target: str,
    optimizations: List[str],
    opset: Any,
    sample_batch: Any,
) -> onnx.ModelProto:
    """
    Exports the torch model to the specified deployment target

    :param model: the torch model to export
    :param deployment_target: the deployment target to export to
    :param sample_batch: a sample batch of data to use for the model
    :return: onnx model exported for the specified deployment target
    """
    (
        disable_bn_fusing,
        use_qlinear_conv,
        use_qlinear_matmul,
        skip_input_quantize,
    ) = setup_optimization_flags(model, optimizations)

    # torch -> onnx
    exporter = TorchToONNX(
        sample_batch=sample_batch, disable_bn_fusing=disable_bn_fusing, opset=opset
    )
    onnx_model = exporter.transform(model)

    if deployment_target == "onnx":
        # nothing to do, already in onnx format
        _LOGGER.info("Exported model to deployment target: onnx")

    elif deployment_target == "deepsparse":
        # onnx -> deepsparse
        exporter = ONNXToDeepsparse(
            use_qlinear_conv=use_qlinear_conv,
            use_qlinear_matmul=use_qlinear_matmul,
            skip_input_quantize=skip_input_quantize,
        )
        onnx_model = exporter.apply(model)
        _LOGGER.info("Exported model to deployment target: deepsparse")

    elif deployment_target == "tensorrt":
        # onnx -> tensorrt
        raise NotImplementedError("tensorrt optimizations not yet implemented")

    else:
        raise ValueError(
            f"Unknown deployment target {deployment_target}. Please choose from ['onnx', 'deepsparse', 'tensorrt']"
        )
    return onnx_model


def setup_optimization_flags(
    model: Module, optimizations: List[str]
) -> Tuple[bool, bool, bool, bool]:
    convert_qat = "convert_qat" in optimizations
    disable_bn_fusing = "disable_bn_fusing" in optimizations
    use_qlinear_conv = (
        hasattr(model, "export_with_qlinearconv")
        and (model.export_with_qlinearconv)
        and convert_qat
    )
    use_qlinear_matmul = (
        hasattr(model, "export_with_qlinearmatmul")
        and (model.export_with_qlinearmatmul)
        and convert_qat
    )
    skip_input_quantize = "skip_input_quantize" in optimizations
    return disable_bn_fusing, use_qlinear_conv, use_qlinear_matmul, skip_input_quantize


def validate_input_optimizations(optimizations: Union[List[str], str, None]) -> Set:
    if optimizations is not None:
        optimizations = (
            [optimizations] if isinstance(optimizations, str) else optimizations
        )
        for optim in optimizations:
            if optim not in AVAILABLE_OPTIMIZATIONS:
                raise ValueError(
                    f"Unknown optimization {optim}. Please choose from {AVAILABLE_OPTIMIZATIONS}"
                )
        return set(optimizations)
    return set()


def apply_kv_cache_injection(onnx_model: onnx.ModelProto):
    model = KeyValueCacheInjector().apply(onnx_model)
