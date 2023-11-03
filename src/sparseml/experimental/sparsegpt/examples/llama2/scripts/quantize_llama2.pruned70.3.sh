#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

BRANCH=$(git branch --show-current)
echo "Branch: $BRANCH"
if [ "$BRANCH" != "tuan/exp_sparsegpt_llama2_gsm8k_03" ]; then
    echo "Expected branch tuan/exp_sparsegpt_llama2_gsm8k_03"
    exit 1
fi

SP=70

ROOT=$HOME/src/neuralmagic/sparseml/src/sparseml/experimental/sparsegpt
RECIPE_DIR=/home/tuan/src/neuralmagic/sparseml/src/sparseml/experimental/sparsegpt/examples/llama2/recipes/llama-2-7b_pruned$SP

DATASET=gsm8k

SRC_MODEL_DIR=/network/tuan/models/llama/GSM8K/base_finetuned_pruned
SRC_MODEL_NAME=llama-2-7b_pruned$SP
SRC_MODEL=$SRC_MODEL_DIR/$SRC_MODEL_NAME

#for RECIPE_NAME in llama-2-7b_quant.bmm.ch llama-2-7b_quant.bmm.skip10.ch llama-2-7b_quant.ch llama-2-7b_quant.skipbmm.skip10.ch
for RECIPE_NAME in llama-2-7b_quant.W8.ch
do
    RECIPE=$RECIPE_DIR/$RECIPE_NAME.md
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
done
