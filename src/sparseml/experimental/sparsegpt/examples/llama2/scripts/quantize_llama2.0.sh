#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

BRANCH=$(git branch --show-current)
echo "Branch: $BRANCH"
if [ "$BRANCH" != "tuan/exp_sparsegpt_llama2_gsm8k_02" ]; then
    echo "Expected branch tuan/exp_sparsegpt_llama2_gsm8k_02"
    exit 1
fi

ROOT=$HOME/src/neuralmagic/sparseml/src/sparseml/experimental/sparsegpt
RECIPE_DIR=/home/tuan/src/neuralmagic/sparseml/src/sparseml/experimental/sparsegpt/examples/llama2/recipes

RECIPE_NAME=llama-2-7b_quant.bmm_v2
RECIPE=$RECIPE_DIR/$RECIPE_NAME.md

DATASET=gsm8k

SRC_MODEL_DIR=/network/tuan/models/llama/GSM8K/base_finetuned_pruned
SRC_MODEL_NAME=llama-2-7b_pruned60

# SRC_MODEL_DIR=/data/models/tuan/llama/potential
# SRC_MODEL_NAME=sparsegpt@@@Llama-2-7b-hf@gsm8k@lr3e-5@B16@W0.1@ep2@GPUs4@@@gsm8k@SP0.6@ID4982

SRC_MODEL=$SRC_MODEL_DIR/$SRC_MODEL_NAME

ID=$RANDOM

DST_MODEL_DIR=/network/tuan/models/llama/GSM8K/ongoing
DST_MODEL_NAME=$SRC_MODEL_NAME@$DATASET@$RECIPE_NAME@ID$ID
DST_MODEL=$DST_MODEL_DIR/$DST_MODEL_NAME

OBSERVER_BATCHES=128

python $ROOT/main.py $SRC_MODEL $DATASET \
       --data-sequence-length 512 \
       --recipe $RECIPE \
       --observer-batches $OBSERVER_BATCHES \
       --save $DST_MODEL

cp "$0" $DST_MODEL/command.sh
