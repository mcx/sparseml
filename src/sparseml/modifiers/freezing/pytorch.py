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

from sparseml.core import ModelParameterizedLayer
from sparseml.core.state import Event, State
from sparseml.modifiers.freezing import GradientFreezeModifier


__all__ = ["GradientFreezeModifierPyTorch"]


class GradientFreezeModifierPyTorch(GradientFreezeModifier):
    frozen_params_: Dict[str, ModelParameterizedLayer] = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        self.frozen_params_ = state.model.get_layers_params(self.targets)
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        for _, param in self.frozen_params_.items():
            param.param.requires_grad = False

    def on_end(self, state: State, event: Event, **kwargs):
        for name, param in self.frozen_params_.items():
            param.requires_grad = True

    def on_finalize(self, state: State, **kwargs) -> bool:
        self.on_end(state, None, **kwargs)
        return True
