#/bin/bash

export CUDA_VISIBLE_DEVICES=0
ROOT=$HOME/src/neuralmagic/lm-evaluation-harness

MODEL_ORG_OR_DIR=/data/models/tuan/llama/ongoing
MODEL_NAME=sparsegpt@@@Llama-2-7b-hf@gsm8k@lr3e-5@B16@W0.1@ep2@GPUs4@@@gsm8k@SP0.7@ID13274

MODEL_ORG_OR_DIR=/network/tuan/models/llama/GSM8K/ongoing
MODEL_NAME=llama-2-7b_pruned60@gsm8k@llama-2-7b_quant@ID25670

MODEL=$MODEL_ORG_OR_DIR/$MODEL_NAME

TASK=gsm8k

SHOTS=0

BATCH=16

python $ROOT/main.py \
       --model hf-causal-experimental \
       --model_args pretrained=$MODEL,use_accelerate=False,dtype=bfloat16 \
       --tasks $TASK \
       --num_fewshot=$SHOTS \
       --batch_size=$BATCH \
       --write_out \
       --output_base_path $HOME/models/test_accuracy_$MODEL_$TASK \
       --device cuda
