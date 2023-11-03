#!/bin/bash

ROOT=/network/tuan/models/llama/GSM8K/gitlab
M=$ROOT/llama-2-7b_base@gsm8k@llama-2-7b_quant.skipbmm.skip10.ch@ID20841
M=$ROOT/llama-2-7b_pruned50@gsm8k@llama-2-7b_quant.skipbmm.skip10.ch@ID3096

S=512

for model_name in llama-2-7b_base@gsm8k@llama-2-7b_quant.skipbmm.skip10.ch@ID20841 llama-2-7b_pruned50@gsm8k@llama-2-7b_quant.skipbmm.skip10.ch@ID3096 llama-2-7b_pruned60@gsm8k@llama-2-7b_quant.skipbmm.skip10.ch@ID19414 llama-2-7b_pruned80@gsm8k@llama-2-7b_quant.skipbmm.skip10.ch@ID4312
do
    M=$ROOT/$model_name
    echo "===========================\n"
    echo "Exporting to ONNX: $M\n"
    rm -r $ROOT/deployment
    sparseml.transformers.export_onnx --model_path $M --sequence_length $S --task text-generation
    echo "Moving deployment folder"
    mv $ROOT/deployment $M
done
