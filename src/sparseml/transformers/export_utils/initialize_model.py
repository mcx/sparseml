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

from pathlib import Path
from typing import Union, Any
from transformers import AutoConfig, AutoTokenizer
from sparseml.transformers.utils import SparseAutoModel
from src.sparseml.transformers.export_utils.tasks import get_aliases, TaskNames

__all__ = ["initialize_model_for_export"]


def initialize_model_for_export(
    model_path: Union[str, Path],
    sequence_length: int,
    task: str,
    trust_remote_code: bool = False,
    **config_args,
):
    # lets also create trainer here but lets not return it

    config = initialize_config(model_path, trust_remote_code, **config_args)
    tokenizer = initialize_tokenizer(model_path, sequence_length, task)
    model = load_task_model(task, model_path, config, trust_remote_code)
    model.train()
    return model, config, tokenizer


def load_task_model(
    task: str, model_path: str, config: Any, trust_remote_code: bool = False
) -> "Module":

    if task in get_aliases(task=TaskNames.MASKED_LANGUAGE_MODELLING):
        return SparseAutoModel.masked_language_modeling_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
            trust_remote_code=trust_remote_code,
        )

    if task in get_aliases(tasks=TaskNames.QUESTION_ANSWERING):
        return SparseAutoModel.question_answering_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
            trust_remote_code=trust_remote_code,
        )

    if task in get_aliases(
        tasks=[
            TaskNames.SEQUENCE_CLASSIFICATION,
            TaskNames.GLUE,
            TaskNames.SENTIMENT_ANALYSIS,
            TaskNames.TEXT_CLASSIFICATION,
        ]
    ):
        return SparseAutoModel.text_classification_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
            trust_remote_code=trust_remote_code,
        )

    if task in get_aliases(tasks=[TaskNames.TOKEN_CLASSIFICATION, TaskNames.NER]):
        return SparseAutoModel.token_classification_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
            trust_remote_code=trust_remote_code,
        )

    if task in get_aliases(tasks=TaskNames.TEXT_GENERATION):
        return SparseAutoModel.text_generation_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
            trust_remote_code=trust_remote_code,
        )

    raise ValueError(f"Unrecognized task: {task}")


def initialize_config(
    model_path: Union[str, Path], trust_remote_code: bool = False, **config_args
):
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        **config_args,
    )
    return config


def initialize_tokenizer(
    model_path: Union[str, Path], sequence_length: int, task: str
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=sequence_length
    )
    if task == TaskNames.TEXT_GENERATION:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
