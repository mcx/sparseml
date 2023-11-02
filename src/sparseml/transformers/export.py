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

import os
import argparse
import logging
from typing import Optional, Union
from src.sparseml.transformers.export_helpers import \
    create_deployment_folder, \
    validate_structure, \
    validate_correctness,
    create_model_inputs,
    export_sample_inputs_outputs,
    initialize_model_for_export,

MODEL_ONNX_NAME = "model.onnx"

_LOGGER = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = None
    return parser.parse_args()


def export_transformer_to_onnx(model_path: Union[str, Path],
                               onnx_file_name: str,
                               sequence_length: int,
                               finetuning_task: Optional[str] = None):

    onnx_file_path = os.path.join(model_path, onnx_file_name)

    # setup the config args for the model
    config_args = {}
    if finetuning_task:
        config_args.update(dict(finetuning_task=finetuning_task))

    # initialize model to export
    _LOGGER.info("Initializing model for export...")

    # Ben: not for export, but in general
    # (swapping all the instances where create a model to use
    # this helper function)
    # Ben: move checkpoint recipy application to automodelphase
    model = initialize_model_for_export(model_path=model_path,
                                        sequence_length=sequence_length,
                                        task=task,
                                        trust_remote_code=trust_remote_code,
                                        **{"finetuning_task": finetuning_task} if finetuning_task else {})
    _LOGGER.info(f"Initialized model for export")

    # initialize trainer to export

    _LOGGER.info("Initializing trainer for export...")
    trainer, recipy_applied = initialize_trainer_for_export()
    _LOGGER.info("Initialized trainer for export")

    # create inputs for the model
    _LOGGER.info("Creating model inputs...")
    inputs = create_model_inputs()
    _LOGGER.info(f"Created model inputs with shape {inputs}.shape")

    # export the model to onnx
    _LOGGER.info("Exporting model to ONNX...")
    export_onnx(model = model,
                inputs = inputs
    )
    _LOGGER.info(f"ONNX exported to {onnx_file_path}")

    # apply the required onnx graph optimizations
    _LOGGER.info("Applying ONNX graph optimizations...")
    apply_onnx_graph_optimizations(onnx_file_path, graph_optimizations)
    _LOGGER.info(f"ONNX graph optimizations: {graph_optimizations} applied")

    # export sample inputs/outputs
    _LOGGER.info(f"Exporting {num_export_samples} sample inputs/outputs...")
    sample_inputs_outputs_dir = export_sample_inputs_outputs()
    _LOGGER.info(f"Exported {num_export_samples} sample inputs/outputs to {sample_inputs_outputs_dir}")

    return onnx_file_path


def export(
    model,
    model_args=None,
    model_path: Union[str, Path],
    onnx_file_name: Optional[str] = None,
    device="auto",
    integration,
    deployment_target="deepsparse",
    export_opset=None,
    samples=None,
    batch_size=None,
    input_shape=None,
    single_graph_file=False,
    graph_optimizations="all",
    validate_structure=True,
    validate_correctness=True,
    **kwargs
):
    integration = integration or kwargs.get("task")
    onnx_model_file_name = onnx_file_name or MODEL_ONNX_NAME

    # given the arguments, export the model to onnx
    export_transformer_to_onnx()

    # create the final deployment folder that consolidates the
    # exported onnx model and all the necessary files for deployment
    deployment_folder_dir = create_deployment_folder(training_directory=model_path,
                                                     onnx_file_name=onnx_file_name)
    if validate_structure:
        validate_structure(deployment_folder_dir)

    if validate_correctness:
        validate_correctness(deployment_folder_dir)

def main():
    # TODO: Eventually, use some nice CLI library like click
    args = parse_args()
    export(
        model = args.model_path,
        model_args = args.model_args,
        path = args.path,
        device = args.device,
        integration = args.integration,
        deployment_target = args.deployment_target,
        export_opset = args.export_opset,
        samples = args.samples,
        batch_size = args.batch_size,
        input_shape = args.input_shape,
        single_graph_file = args.single_graph_file,
        graph_optimizations = args.graph_optimizations,
        validate_structure = args.validate_structure,
        validate_correctness = args.validate_correctness,
    )

if __name__ == "__main__":
    main()