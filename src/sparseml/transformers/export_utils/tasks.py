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

from typing import List, Union
from dataclasses import dataclass

__all__ = ["TaskNames", "get_aliases"]


@dataclass(frozen=True)
class TaskNames:
    MASKED_LANGUAGE_MODELLING: str = "masked-language-modeling"
    QUESTION_ANSWERING: str = "question-answering"
    SEQUENCE_CLASSIFICATION: str = "sequence-classification"
    SENTIMENT_ANALYSIS: str = "sentiment-analysis"
    TOKEN_CLASSIFICATION: str = "token-classification"
    NER: str = "ner"
    TEXT_GENERATION: str = "text-generation"


def get_aliases(
    tasks: Union[str, List[str]], task_names=TaskNames()
) -> List[Union[str, None]]:
    aliases = []
    if isinstance(tasks, str):
        tasks = [tasks]
    for task in tasks:
        if task in task_names.__dict__.values():
            aliases.append(task)
        alias = {
            task_names.MASKED_LANGUAGE_MODELLING: "mlm",
            task_names.QUESTION_ANSWERING: "qa",
        }.get(task)
        if alias:
            aliases.append(alias)
    return aliases


get_aliases("masked-language-modeling")
