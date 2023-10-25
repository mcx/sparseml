#/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3
ROOT=$HOME/src/EleutherAI/lm-evaluation-harness

MODEL_ORG_OR_DIR=/data/models/tuan/llama/ongoing
MODEL_NAME=sparsegpt@@@Llama-2-7b-hf@gsm8k@lr3e-5@B16@W0.1@ep2@GPUs4@@@gsm8k@SP0.5@ID17912
MODEL_NAME=sparsegpt@@@Llama-2-7b-hf@gsm8k@lr3e-5@B16@W0.1@ep2@GPUs4@@@gsm8k@SP0.8@ID16456
MODEL_NAME=sparsegpt@@@Llama-2-7b-hf@gsm8k@lr3e-5@B16@W0.1@ep2@GPUs4@@@gsm8k@SP0.6@ID4982
MODEL_NAME=sparsegpt@@@Llama-2-7b-hf@gsm8k@lr3e-5@B16@W0.1@ep2@GPUs4@@@gsm8k@SP0.7@ID13274

SRC_MODEL_DIR=/network/tuan/models/llama/GSM8K/base_finetuned_pruned
SRC_MODEL_NAME=llama-2-7b_pruned60

# MODEL_ORG_OR_DIR=$HOME/models
# MODEL_NAME=Llama-2-7b-hf

MODEL=$MODEL_ORG_OR_DIR/$MODEL_NAME

TASK=gsm8k

SHOTS=0

BATCH=16

python $ROOT/main.py \
       --model hf-causal-experimental \
       --model_args pretrained=$MODEL,use_accelerate=True,dtype=bfloat16 \
       --tasks $TASK \
       --num_fewshot=$SHOTS \
       --batch_size=$BATCH \
       --write_out \
       --output_base_path $HOME/models/test_accuracy_$MODEL_$TASK \
       --device cuda
