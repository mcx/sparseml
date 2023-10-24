import copy
from datasets import load_dataset
from transformers import LlamaTokenizer

from transformers import LlamaForCausalLM
import torch
import os
from sparseml.experimental.sparsegpt.llama2 import cache_attention_inputs
from sparseml.experimental.sparsegpt.utils import execute_offloaded_module
import numpy

SEED = 2023
NSAMPLES = 128

SRC_MODEL_DIR = "/network/tuan/models/llama/GSM8K/base_finetuned_pruned"
SRC_MODEL_NAME = "llama-2-7b_pruned60"
model_name_or_path = os.path.join(SRC_MODEL_DIR, SRC_MODEL_NAME)

path = os.path.join(SRC_MODEL_DIR, SRC_MODEL_NAME, "stats")
if not os.path.exists(path):
    os.makedirs(path)
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
            example = example[: max_seq_len]
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
            "attention_mask":example_mask,
        }

    dataset_train = dataset_train.map(
        lambda sample: process_sample(sample),
        batched=False,
        remove_columns=list(dataset_train.features),
    )

    return dataset_train


def save_stats(name):
    print(f"Collecting stats {name}...\n")
    data = [numpy.abs(i.cpu().numpy().reshape(-1, i.shape[-1])) for i in inputs[name]]
    data = numpy.concatenate(data)

    # Record stats per channel
    mean = numpy.mean(data, axis=0)
    std = numpy.std(data, axis=0)
    median = numpy.median(data, axis=0)
    min_val = numpy.min(data, axis=0)
    max_val = numpy.max(data, axis=0)
    p25 = numpy.percentile(data, 25, axis=0)
    p75 = numpy.percentile(data, 75, axis=0)
    numpy.savez(
        os.path.join(path, name + ".npz"),
        mean=mean,
        std=std,
        median=median,
        min_val=min_val,
        max_val=max_val,
        p25=p25,
        p75=p75,
    )

    # Record stats per tensor
    mean = numpy.mean(data.flatten())
    std = numpy.std(data.flatten())
    median = numpy.median(data.flatten())
    min_val = numpy.min(data.flatten())
    max_val = numpy.max(data.flatten())
    p25 = numpy.percentile(data.flatten(), 25)
    p75 = numpy.percentile(data.flatten(), 75)
   
    numpy.savez(
        os.path.join(path, name + "_scalar.npz"),
        mean=mean,
        std=std,
        median=median,
        min_val=min_val,
        max_val=max_val,
        p25=p25,
        p75=p75,
    )


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
            save_stats(n)
            del inputs[n]

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
    previous_names = list(inputs.keys())
    for n in previous_names:
        save_stats(n)
        del inputs[n]

    return logits


model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto")
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, torch_dtype="auto")
    
handles = []
inputs = {}

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"Register hook for {name}\n")
        handles.append(module.register_forward_hook(extract_inputs(name)))

model.eval()

dataset = get_gsm8k()

with torch.no_grad():
    llama2_forward(model, dataset, device)


    
