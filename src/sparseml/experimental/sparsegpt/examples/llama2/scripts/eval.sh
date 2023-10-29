#/bin/bash

export CUDA_VISIBLE_DEVICES=4
ROOT=$HOME/src/neuralmagic/lm-evaluation-harness

MODEL_ORG_OR_DIR=/network/tuan/models/llama/GSM8K/ongoing
MODEL_NAME=llama-2-7b_pruned60@gsm8k@llama-2-7b_quant.group0-4@ID30304
MODEL=$MODEL_ORG_OR_DIR/$MODEL_NAME

TASK=gsm8k

SHOTS=0

BATCH=8

python $ROOT/main.py \
       --model hf-causal-experimental \
       --model_args pretrained=$MODEL,use_accelerate=False,dtype=bfloat16 \
       --tasks $TASK \
       --num_fewshot=$SHOTS \
       --batch_size=$BATCH \
       --write_out \
       --output_base_path $HOME/models/test_accuracy_$MODEL_$TASK \
       --device cuda
