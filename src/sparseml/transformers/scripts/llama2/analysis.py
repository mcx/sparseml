import collections
import copy
import os

import numpy
import pandas as pd
import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama import QuantizableMatMul

from sparseml.experimental.sparsegpt.llama2 import cache_attention_inputs
from sparseml.experimental.sparsegpt.utils import execute_offloaded_module


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

PADDING = False

SEED = 2023
NSAMPLES = 1319

SRC_MODEL_DIR = "/network/tuan/models/llama/GSM8K/base_finetuned_pruned"
SRC_MODEL_NAME = "llama-2-7b_pruned60"
#model_name_or_path = os.path.join(SRC_MODEL_DIR, SRC_MODEL_NAME)
model_name_or_path = "/network/tuan/src/neuralmagic/ml-experiments/nlg-text_generation/gsm8k-llama2_7b-oneshot_sparse_finetune_sparsegpt/pruned80/training"

model_name_or_path = "/network/tuan/src/neuralmagic/ml-experiments/nlg-text_generation/gsm8k-llama2_7b-base/dense/training"

model_name_or_path = "/network/tuan/models/llama/GSM8K/clip_softmax/Llama-2-7b-hf@gsm8k@lr3e-5@B16@GrAcc1@W0.1@ep2@GPUs4@ClipSM-0.001@ID25986/hf"

model_name_or_path = "/network/tuan/models/llama/GSM8K/clip_softmax/Llama-2-7b-hf@gsm8k@lr3e-5@B16@GrAcc1@W0.1@ep2@GPUs4@ClipSM-0.02@ID29890/hf"


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


def save_stats_v2(name):
    print(f"Collecting stats {name}...\n")
    sum_max = 0.0
    for ss in inputs[name]:
        m = torch.max(torch.abs(ss))
        sum_max += m
    abs_max = sum_max / len(inputs[name])
    stats_dict = collections.OrderedDict(
        {
            "module": name,
            "avg_max_inf_norm": abs_max.cpu().numpy(),
        }
    )

    torch.cuda.empty_cache()
    return stats_dict


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
            if n.find(MODULE_TYPE) < 0:
                continue
            stats = save_stats_v2(n)
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

root = "/network/tuan/models/llama/GSM8K/clip_softmax"

for f in [
    "Llama-2-7b-hf@gsm8k@lr3e-5@B16@GrAcc1@W0.1@ep2@GPUs4@ClipSM-0.001@ID25986"
]:

    model_name_or_path = os.path.join(root, f, "hf")
    print(f"=== {f} ===========\n")
    stats_path = os.path.join(model_name_or_path, f"stats_seqlen512_padding{PADDING}")
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
    device = "cuda:7"

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

        dataset = get_gsm8k()

        with torch.no_grad():
            logits, stats_dict = llama2_forward(model, dataset, device)

        df = pd.DataFrame.from_dict(stats_dict)
        fname = f"{MODULE_TYPE}_input_tensors.csv" if PADDING else f"{MODULE_TYPE}_input_tensors_no_padding.csv"
        df.to_csv(os.path.join(stats_path, fname), index=False)

        model = None
        tokenizer = None
        torch.cuda.empty_cache()
        print(f"Done {MODULE_TYPE}")
