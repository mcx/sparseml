import collections
import copy
import os
from typing import List

import numpy
import pandas as pd
import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama import QuantizableMatMul

from sparseml.experimental.sparsegpt.llama2 import cache_attention_inputs
from sparseml.experimental.sparsegpt.utils import execute_offloaded_module


PADDING = False

MODULE_TYPE = "down_proj"
assert MODULE_TYPE in [
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "attn_weights_matmul",
    "attn_output_matmul",
]



SAMPLE_INDICES = [0]
SEED = 2023
NSAMPLES = 64

model_name_or_path = "/network/tuan/models/llama/GSM8K/clip_softmax/Llama-2-7b-hf@gsm8k@lr3e-5@B16@GrAcc1@W0.1@ep2@GPUs4@ClipSM_OFF@ID18250/hf"

stats_path = os.path.join(model_name_or_path, f"stats_seqlen512_padding{PADDING}")
if not os.path.exists(stats_path):
    os.makedirs(stats_path)
device = "cuda:7"


def get_gsm8k(sample_indices: List[int] = None, nsamples: int = NSAMPLES, seed: int = SEED):
    dataset_train = load_dataset("gsm8k", "main", split="train")
    if sample_indices is not None and len(sample_indices) > 0:
        print(f"Limitting dataset to indices {sample_indices}\n")
        dataset_train = dataset_train.select(sample_indices)
    else:
        numpy.random.seed(seed)
        rand_indices = numpy.random.randint(len(dataset_train), size=nsamples)
        print(f"Randomly select {nsamples} data points")
        dataset_train = dataset_train.select(rand_indices)

    def process_sample(sample):
        IGNORE_INDEX = -100
        prompt = f"Question: {{question}}.\nAnswer:".format(
            question=sample["question"],
        )
        example = f"Question: {{question}}.\nAnswer: {{answer}}{{eos_token}}".format(
            question=sample["question"],
            answer=sample["answer"],
            eos_token=tokenizer.eos_token,
        )
        prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)
        example = torch.tensor(tokenizer.encode(example), dtype=torch.int64)
        max_seq_len = 512
        if PADDING:
            padding = max_seq_len - example.shape[0]
            if padding > 0:
                example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example = example[:max_seq_len]
        else:
            example = example[:max_seq_len]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        if PADDING:
            res = {
                "input_ids": example,
                "labels": labels,
                "attention_mask": example_mask,
            }
        else:
            res = {
                "input_ids": example,
            }
        return res

    dataset_train = dataset_train.map(
        lambda sample: process_sample(sample),
        batched=False,
        remove_columns=list(dataset_train.features),
    )

    return dataset_train


def save_outlier_stats(name):
    print(f"Counting outliers {name}...\n")
    abs_data = torch.mean(torch.abs(torch.cat(inputs[name])), 0).cpu().numpy()
    mean_abs = numpy.mean(abs_data)
    std_abs = numpy.std(abs_data)
    delta_abs = numpy.abs(abs_data - mean_abs)
    outliers = delta_abs > 6 * std_abs
    outliers_per_token = numpy.sum(outliers, axis=1)
    outliers_per_feature = numpy.sum(outliers, axis=0)
    per_token_outliers = collections.OrderedDict(
        {"module": name}
    )
    per_token_outliers.update(collections.OrderedDict(
        {
            f"token_{k}": outliers_per_token[k] for k in range(len(outliers_per_token))
        }
    ))
    per_feature_outliers = collections.OrderedDict(
        {"module": name}
    )
    per_feature_outliers = collections.OrderedDict(
        {
            f"feat_{k}": outliers_per_feature[k] for k in range(len(outliers_per_feature))
        }
    )

    del abs_data
    del mean_abs
    del std_abs
    del delta_abs
    del outliers
    del outliers_per_token
    del outliers_per_feature
    torch.cuda.empty_cache()
    return per_token_outliers, per_feature_outliers


def save_outlier_stats_v2(name):
    print(f"Counting outliers {name}...\n")
    assert len(inputs[name]) == 1
    data = inputs[name][0].cpu().numpy()
    mean = numpy.mean(data)
    std = numpy.std(data)
    delta = numpy.abs(data - mean)
    outliers = delta > 6 * std
    outliers=numpy.squeeze(outliers, axis=0)
    outliers_per_token = numpy.sum(outliers, axis=1)
    outliers_per_feature = numpy.sum(outliers, axis=0)
    per_token_outliers = collections.OrderedDict(
        {"module": name}
    )
    per_token_outliers.update(collections.OrderedDict(
        {
            f"token_{k}": outliers_per_token[k] for k in range(len(outliers_per_token))
        }
    ))
    per_feature_outliers = collections.OrderedDict(
        {"module": name}
    )
    per_feature_outliers = collections.OrderedDict(
        {
            f"feat_{k}": outliers_per_feature[k] for k in range(len(outliers_per_feature))
        }
    )

    torch.cuda.empty_cache()
    return per_token_outliers, per_feature_outliers


def extract_inputs_linear_module(name):
    def tmp(_, inp, out):
        if name in inputs:
            inputs[name].append(inp[0])
        else:
            inputs[name] = [inp[0]]

    return tmp


def extract_inputs_bmm_module(name):
    def tmp(_, inp, out):
        assert len(inp) == 2
        for k in range(2):
            inp_name = f"{name}_inp_{k}"
            if inp_name in inputs:
                inputs[inp_name].append(inp[k])
            else:
                inputs[inp_name] = [inp[k]]

    return tmp


def _accumulate_stats(stats, accumulated_stats):
    if accumulated_stats is None:
        accumulated_stats = {k: [stats[k]] for k in stats.keys()}
    else:
        assert list(accumulated_stats.keys()) == list(stats.keys())
        for k in accumulated_stats.keys():
            accumulated_stats[k].append(stats[k])
    return accumulated_stats


def llama2_forward(model, data_loader, device, nsamples=None):
    # Catch attention mask
    cached_inputs = cache_attention_inputs(model, data_loader, device, nsamples)
    buffer = [b[0] for b in cached_inputs.pop("inputs")]
    per_token_outliers = None
    per_feat_outliers = None
    for layer in model.model.layers:
        buffer = execute_offloaded_module(
            layer,
            buffer,
            device,
            cached_inputs=cached_inputs,
            use_cache=False,
        )
        buffer = [b[0] for b in buffer]
        previous_names = list(inputs.keys())
        for n in previous_names:
            if n.find(MODULE_TYPE) < 0:
                continue
            per_token, per_feat = save_outlier_stats_v2(n)
            del inputs[n]
            torch.cuda.empty_cache()

            per_token_outliers = _accumulate_stats(per_token, per_token_outliers)
            per_feat_outliers = _accumulate_stats(per_feat, per_feat_outliers)

    del cached_inputs
    torch.cuda.empty_cache()

    buffer = execute_offloaded_module(
        model.model.norm,
        buffer,
        device,
    )
    logits = execute_offloaded_module(
        model.lm_head,
        buffer,
        device,
    )

    return logits, (per_token_outliers, per_feat_outliers)


for MODULE_TYPE in [
    "down_proj",
]:
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto")
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, torch_dtype="auto")
    handles = []
    inputs = {}

    for name, module in model.named_modules():
        if not name.endswith(MODULE_TYPE):
            continue
        if isinstance(module, torch.nn.Linear):
            print(f"Register hook for {name}\n")
            handles.append(module.register_forward_hook(extract_inputs_linear_module(name)))
        elif isinstance(module, QuantizableMatMul):
            print(f"Register hook for QuantizableMatMul {name}\n")
            handles.append(module.register_forward_hook(extract_inputs_bmm_module(name)))

    model.eval()

    dataset = get_gsm8k(sample_indices=SAMPLE_INDICES, nsamples=NSAMPLES, seed=SEED)

    with torch.no_grad():
        logits, (per_token_outliers, per_feat_outliers) = llama2_forward(model, dataset, device)

    assert len(SAMPLE_INDICES) == 1
    sample_id = SAMPLE_INDICES[0]
    df = pd.DataFrame.from_dict(per_token_outliers)
    fname = f"{MODULE_TYPE}_per_token_outliers_sample{sample_id}.csv" if PADDING else f"{MODULE_TYPE}_per_token_outliers_sample{sample_id}_no_padding.csv"
    df.to_csv(os.path.join(stats_path, fname), index=False)

    df = pd.DataFrame.from_dict(per_feat_outliers)
    fname = f"{MODULE_TYPE}_per_feat_outliers_sample{sample_id}.csv" if PADDING else f"{MODULE_TYPE}_per_feat_outliers_sample{sample_id}_no_padding.csv"
    df.to_csv(os.path.join(stats_path, fname), index=False)

    torch.cuda.empty_cache()
    print(f"Done {MODULE_TYPE}")
