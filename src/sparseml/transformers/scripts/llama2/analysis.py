import collections
import copy
import os

import numpy
import pandas as pd
import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

from sparseml.experimental.sparsegpt.llama2 import cache_attention_inputs
from sparseml.experimental.sparsegpt.utils import execute_offloaded_module


MODULE_TYPE = "gate_proj"
assert MODULE_TYPE in ["k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

SEED = 2023
NSAMPLES = 64

SRC_MODEL_DIR = "/network/tuan/models/llama/GSM8K/base_finetuned_pruned"
SRC_MODEL_NAME = "llama-2-7b_pruned60"
model_name_or_path = os.path.join(SRC_MODEL_DIR, SRC_MODEL_NAME)

stats_path = os.path.join(SRC_MODEL_DIR, SRC_MODEL_NAME, "stats")
if not os.path.exists(stats_path):
    os.makedirs(stats_path)
device = "cuda:1"


def get_gsm8k(nsamples: int = NSAMPLES, seed: int = SEED):
    dataset_train = load_dataset("gsm8k", "main", split="train")
    numpy.random.seed(seed)
    rand_indices = numpy.random.randint(len(dataset_train), size=nsamples)
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
        max_seq_len = 1024
        padding = max_seq_len - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:max_seq_len]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

    dataset_train = dataset_train.map(
        lambda sample: process_sample(sample),
        batched=False,
        remove_columns=list(dataset_train.features),
    )

    return dataset_train


def save_stats(name):
    print(f"Collecting stats {name}...\n")
    abs_data = torch.mean(torch.abs(torch.cat(inputs[name])), 0).cpu().numpy()

    stats_dict = collections.OrderedDict(
        {
            "module": name,
            "mean_abs": numpy.mean(abs_data),
            "std_abs": numpy.std(abs_data),
            "max_abs": numpy.max(abs_data),
            "q50_abs": numpy.median(abs_data),
            "q25_abs": numpy.quantile(abs_data, 0.25),
            "q75_abs": numpy.quantile(abs_data, 0.75),
        }
    )

    del abs_data
    torch.cuda.empty_cache()
    return stats_dict


def extract_inputs(name):
    def tmp(_, inp, out):
        if name in inputs:
            inputs[name].append(inp[0])
        else:
            inputs[name] = [inp[0]]

    return tmp


def llama2_forward(model, data_loader, device, nsamples=None):
    # Catch attention mask
    cached_inputs = cache_attention_inputs(model, data_loader, device, nsamples)
    buffer = [b[0] for b in cached_inputs.pop("inputs")]
    stats_dict = None
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
            if not n.endswith(MODULE_TYPE):
                continue
            stats = save_stats(n)
            del inputs[n]
            torch.cuda.empty_cache()
            if stats_dict is None:
                stats_dict = {k: [stats[k]] for k in stats.keys()}
            else:
                assert list(stats_dict.keys()) == list(stats.keys())
                for k in stats_dict.keys():
                    stats_dict[k].append(stats[k])

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

    return logits, stats_dict


model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto")
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, torch_dtype="auto")

handles = []
inputs = {}

for name, module in model.named_modules():
    print(f"{name}\n")
    if isinstance(module, torch.nn.Linear) and name.endswith(MODULE_TYPE):
        print(f"Register hook for {name}\n")
        handles.append(module.register_forward_hook(extract_inputs(name)))

model.eval()

dataset = get_gsm8k()

with torch.no_grad():
    logits, stats_dict = llama2_forward(model, dataset, device)

df = pd.DataFrame.from_dict(stats_dict)
df.to_csv(os.path.join(stats_path, f"{MODULE_TYPE}_input_tensors.csv"), index=False)

print("Done")
