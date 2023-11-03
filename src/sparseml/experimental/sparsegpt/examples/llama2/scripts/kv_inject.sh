#!/bin/bash

ROOT=$HOME/models/gitlab


for model_name in llama2-7b-gsm8k_llama2_pretrain-pruned50_quantized
do
    M=$ROOT/$model_name/deployment
    echo "KV injection: $M\n"
    python ./examples/llama2/scripts/kv_inject.py --input-file $M/model.onnx --output-file $M/model_cache.onnx
    # mv $M/model.onnx $M/model_nocache.onnx
    # mv $M/model_cache.onnx $M/model.onnx
done
