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


import collections
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm
import torch
import re
from sparseml.core.model import ModifiableModel
from sparseml.core.state import State
from sparseml.modifiers.obcq.base import SparseGPTModifier
from sparseml.modifiers.obcq.utils.helpers import cache_attention_inputs
from sparseml.modifiers.obcq.utils.layer_compressor import LayerCompressor


_LOGGER = logging.getLogger(__name__)

NEW = True

class SparseGPTModifierPyTorch(SparseGPTModifier):
    """
    Pytorch implementation of SparseGPT

    Lifecycle:
        - on_initialize
            - initialize_obcq
                - compressible_layers
            - apply_obcq
                - compress_bottom
                - LayerCompressor.compress
        - on_finalize

    :param model: Pytorch model to perform OBCQ on, in-place
    """

    model: Any = None
    device_: str = "cuda:0"
    layer_prefix_: Optional[str] = None

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Initialize and run the OBCQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        if not self.initialized_structure_:
            self.on_initialize_structure(state, **kwargs)
        if self.quantization_modifier_:
            self.quantization_modifier_.initialize(state, **kwargs)
        device = state.hardware.device
        calibration_dataloader = state.data.calib
        modifiable_model = state.model
        self.initialize_obcq(modifiable_model, calibration_dataloader, device)
        self.apply_obcq(calibration_dataloader)

        return True

    def initialize_obcq(
        self,
        model: "ModifiableModel",
        dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None,
        device: Optional[str] = "cuda:0",
    ):
        """
        Setup for SparseGPT, initialize the the compressible layers of model, and set
        the device

        :param model: PyTorch model to sparsify
        :param device: device to run sparsification on, preferably a GPU
        """
        self.model = model
        self.compressible_layers_ = self.compressible_layers()
        self.layer_prefix_ = model.layer_prefix
        self.model = self.model.model
        self._set_device(device)
        self._infer_mask_block_size()
        if self.sparsity_profile is not None and self.sparsity_profile.lower() == "owl":
            if NEW:
                self.sparsity = self._infer_layer_sparsity_NEW(dataloader, device)
                _LOGGER.info(f">>> Sparsity levels: {self.sparsity}\n")
            else:
                self.sparsity = self._infer_layer_sparsity(dataloader, device)
        self._validate_layerwise_sparsity()

    @torch.no_grad()
    def apply_obcq(
        self, dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None
    ) -> Dict:
        """
        Run OBCQ on the loaded model, using dataloader as calibration data

        :param dataloader: calibration data for OBCQ
        """
        accum_kwargs = {"dataloader": dataloader}

        # Step 0: Pass the calibration data through the (compressed) bottom part of the
        # network, capturing the outputs which will become the inputs to the first
        # decoder layer. Also return attention_mask as part of kwargs
        extras = self.compress_bottom(
            dev=self.device_,
            target_ids=self.target_ids,
            layer_prefix=self.layer_prefix_,
            **accum_kwargs,
        )
        accum_kwargs.update(extras)

        # Step 1: Sequentially prune/quantize decoder layers
        inputs = None
        num_layers = len(self.compressible_layers_)
        idx = -1
        for layer_name, layer in self.compressible_layers_.items():
            idx += 1
            if "outputs" not in accum_kwargs:
                raise RuntimeError(
                    "The 'outputs' key is expected but not found from the "
                    "return of the bottom compressor"
                )
            inputs = accum_kwargs["outputs"]
            layer_sparsity = (
                self.sparsity[idx] if isinstance(self.sparsity, List) else self.sparsity
            )

            _LOGGER.info(
                f"\n===== Compressing layer {idx+1}/{num_layers} ====="
            )
            args = {
                "sparsity": self.sparsity, #layer_sparsity,
                "prunen": self.prunen_,
                "prunem": self.prunem_,
                "blocksize": self.block_size,
                "percdamp": self.dampening_frac,
                "sequential_update": self.sequential_update,
                "quantize": self.quantize,
            }
            layer_compressor = LayerCompressor(self.model, layer_name, layer, idx, inputs, args)

            # Prune/quantize using SparseGPT
            layer_kwargs = layer_compressor.compress(dev=self.device_, **accum_kwargs)
            accum_kwargs.update(layer_kwargs)

    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the observers used by the OBCQ algorithm and set kv-cache configuration

        :param state: un-used, for matching spec of Modifier base class
        """

        if self.quantization_modifier_:
            self.quantization_modifier_.finalize(state, **kwargs)

        return True

    def compress_bottom(
        self,
        dataloader: List = None,
        nsamples: int = None,
        dev: str = "cuda:0",
        target_ids: List[str] = None,
        layer_prefix: Optional[str] = None,
    ) -> Dict:
        """
        Runs calibration data through the bottom part of the network (everything up
        to the first decoder layer) and return the captured outputs

        :param dataloader: calibration data to pass through the model
        :param nsamples: number of samples to use for calibration, or None to use it all
        :param dev: device to use
        :param target_ids: list of keys in model output to cache, NOTE: this argument
            has been deprecated and will be removed in a future release
        :param layer_prefix: name of model attribute that contains the list of layers,
            i.e. model.decoder for OPT or just model for Llama
        :return: outputs from bottom part of network, attention mask, and kv-cache state
        """
        layer_prefix = layer_prefix or self.layer_prefix_
        cached_inputs = cache_attention_inputs(
            model=self.model,
            dataloader=dataloader,
            device=dev,
            nsamples=nsamples,
            target_ids=target_ids,
            layer_prefix=layer_prefix,
        )

        outputs = cached_inputs.pop("inputs")
        outputs = [o[0] for o in outputs]
        cached_inputs.update({"outputs": outputs})
        return cached_inputs

    def _set_device(self, device: str):
        if "cuda" in device and not torch.cuda.is_available():
            self.device_ = "cpu"
        else:
            self.device_ = device

    def _infer_mask_block_size(self):
        """
        Infer the mask block size from the mask structure.
        Parses mask_structure of the form N:M where N, M are integers that
        define a custom block shape; and sets prunen_ and prunem_ accordingly.

        :post-condition: prunen_ and prunem_ are set
        """
        if self.mask_structure is None:
            raise ValueError("mask_structure must be defined")

        self.prunen_, self.prunem_ = list(map(int, self.mask_structure.split(":")))

    def _get_owl_group(self, linear_module_name):
        if self.owl_outlier_granularity == "decoder_layer":
            match = re.search(r"layers.\d+", linear_module_name)
            group = match.group() if match else None
        elif self.owl_outlier_granularity == "linear_module":
            group = linear_module_name
        return group

    def _infer_layer_sparsity(self, calibration_dataloader, dev):
        prev_dev = self.model.device
        self.model.to(dev)
        acts = _get_activations(self.model, calibration_dataloader)
        wanda = []
        names = []
        num_w_perlayer = 0
        import pdb; pdb.set_trace()
        for n, m in self.model.named_modules():
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n:
                print(f"[DEBUG] owl considering = {n} with shape = {m.weight.shape}")
                if "layers.1." in n:
                    num_w_perlayer += 1
                wanda.append(m.weight.abs() * acts[n].unsqueeze(0))
                print(f"wanda score shape = {wanda[-1].shape}")
                names.append(n)
        print(f"[DEBUG] there is {num_w_perlayer} Linear weight matrices per layer")
        perlayer_wanda = [
            torch.cat([item.flatten().cpu() for item in wanda[i : i + num_w_perlayer]])
            for i in range(0, len(wanda), num_w_perlayer)
        ]

        acts = None
        del acts
        del wanda
        self.model.to(prev_dev)
        torch.cuda.empty_cache()

        outlier_ratios = []
        for wanda_layer in perlayer_wanda:
            threshold = torch.mean(wanda_layer) * self.owl_m
            outlier_ratios.append(
                100 * (wanda_layer > threshold).sum().item() / wanda_layer.numel()
            )
        import numpy as np

        outlier_ratios = np.array(outlier_ratios)
# [1.2220837291658233, 1.2492135398746154, 1.478302911155582, 1.9342516370387892, 1.4717121816052057, 1.3769110249731824, 1.5513978473880747, 1.4558930471153457, 1.3137486314526494, 1.2885691588406736, 1.2739295168861824, 1.249576232593912, 1.2503940206735245, 1.1581193597823227, 1.1089843789530542, 0.9560762909409913, 0.8759597422545438, 0.7874884135982533, 0.7319524498183493, 0.6658909852022952, 0.6364891566143135, 0.6336404869593487, 0.613886087051945, 0.5999594772417929, 0.5670468424268337, 0.5980402694464965, 0.5311970883700514, 0.49776907411881677, 0.4965362153522709, 0.44302668596178757, 0.4447037691897061, 0.5180408299895766]


        outlier_ratios = (outlier_ratios - outlier_ratios.min()) * (
            1 / (outlier_ratios.max() - outlier_ratios.min()) * self.owl_lmbda * 2
        )
# array([2.08971032e-02, 2.16248220e-02, 2.77698204e-02, 4.00000000e-02,
#        2.75930334e-02, 2.50501264e-02, 2.97304886e-02, 2.71687075e-02,
#        2.33558846e-02, 2.26804808e-02, 2.22877931e-02, 2.16345507e-02,
#        2.16564867e-02, 1.91813495e-02, 1.78633731e-02, 1.37618300e-02,
#        1.16128169e-02, 9.23969861e-03, 7.75002493e-03, 5.97801959e-03,
#        5.18935713e-03, 5.11294559e-03, 4.58306176e-03, 4.20950015e-03,
#        3.32666527e-03, 4.15802011e-03, 2.36504633e-03, 1.46838713e-03,
#        1.43531744e-03, 0.00000000e+00, 4.49853854e-05, 2.01214831e-03])

        sparsities_per_tflayer = list(
            1 - (outlier_ratios - np.mean(outlier_ratios) + (1 - float(self.sparsity)))
        )
# [0.39318857942475305, 0.39246086062715113, 0.3863158622856062, 0.37408568264560194, 0.38649264928331883, 0.3890355562153295, 0.3843551940127289, 0.38691697519010937, 0.39072979807937325, 0.3914052018162485, 0.3917978895041502, 0.39245113190797876, 0.39242919589949066, 0.3949043331569396, 0.3962223095170798, 0.4003238526920817, 0.4024728657870835, 0.4048459840370805, 0.4063356577107886, 0.40810766305628243, 0.4088963255197625, 0.4089727370538615, 0.4095026208891108, 0.4098761824983126, 0.4107590173728548, 0.40992766253273416, 0.4117206363165532, 0.412617295515478, 0.41265036520803255, 0.41408568264560197, 0.4140406972602191, 0.41207353433830296]

        print(f"[DEBUG] OWL sparsities for sp={self.sparsity} are:")
        sparsities = []
        for j, sp in enumerate(sparsities_per_tflayer):
            print(f"layers.{j} sparsity = {sp}")
            sparsities += [sp] * num_w_perlayer

        return sparsities_per_tflayer

    def _infer_layer_sparsity_NEW(self, calibration_dataloader, dev):
        prev_dev = self.model.device
        self.model.to(dev)
        acts = _get_activations(self.model, calibration_dataloader)

        wanda = collections.defaultdict(list)
        names = []

        # TODO: move this block
        if self.owl_outlier_granularity is None:
            self.owl_outlier_granularity = "decoder_layer"
        if self.owl_outlier_granularity not in ["decoder_layer", "linear_module"]:
            raise ValueError("Unknown OWL granularity")

        for n, m in self.model.named_modules():
            # TODO: check if this module is compressible from recipe
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n:
                group = self._get_owl_group(n)
                wanda[group].append(m.weight.abs() * acts[n].unsqueeze(0))
                names.append(n)

        dd = {}
        for group in tqdm(wanda):
            v = torch.cat([item.flatten().cpu() for item in wanda[group]])
            dd[group] = v

        acts = None
        del acts
        del wanda
        self.model.to(prev_dev)
        torch.cuda.empty_cache()

        wanda = dd

        outlier_ratios = {}
        for group in wanda:
            threshold = torch.mean(wanda[group]) * self.owl_m
            outlier_ratios[group] = (
                100 * (wanda[group] > threshold).sum().item() / wanda[group].numel()
            )
        import numpy as np

        outlier_ratios_arr = np.array([outlier_ratios[k] for k in outlier_ratios])
        for k in outlier_ratios:
            outlier_ratios[k] = (outlier_ratios[k] - outlier_ratios_arr.min()) * (
                1
                / (outlier_ratios_arr.max() - outlier_ratios_arr.min())
                * self.owl_lmbda
                * 2
            )
        outlier_ratios_arr = np.array([outlier_ratios[k] for k in outlier_ratios])
        sparsities = {
            k: 1
            - (
                outlier_ratios[k]
                - np.mean(outlier_ratios_arr)
                + (1 - float(self.sparsity))
            )
            for k in outlier_ratios
        }

        return sparsities


@torch.no_grad()
def _get_activations(model, data_loader, nsamples=128):
    import functools

    model.eval()
    acts = {}

    def save_acts(module, input, name):
        if isinstance(input, tuple):
            input = input[0]
        if name not in acts:
            acts[name] = 1.0 / nsamples * input.detach().pow(2).sum(dim=(0, 1)).sqrt()
        else:
            acts[name] += 1.0 / nsamples * input.detach().pow(2).sum(dim=(0, 1)).sqrt()

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            hooks.append(
                mod.register_forward_pre_hook(functools.partial(save_acts, name=name))
            )
    device = next(model.parameters()).device
    for batch in data_loader:
        batch = batch.to(device)
        model(batch)

    for h in hooks:
        h.remove()

    return acts
