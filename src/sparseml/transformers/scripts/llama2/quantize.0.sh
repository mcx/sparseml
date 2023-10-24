#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ROOT=$HOME/src/neuralmagic/sparseml/src/sparseml/transformers

RECIPE_DIR=$ROOT/recipes/llama2
RECIPE_NAME=example_llama

SRC_MODEL_DIR=/network/tuan/models/llama/GSM8K/base_finetuned_pruned
SRC_MODEL_NAME=llama-2-7b_pruned60
SRC_MODEL=$SRC_MODEL_DIR/$SRC_MODEL_NAME

DATASET=gsm8k

DST_MODEL_DIR=/network/tuan/models/llama/GSM8K/ongoing
DST_MODEL_NAME=$SRC_MODEL_NAME@$DATASET@$RECIPE_NAME
DST_MODEL=$DST_MODEL_DIR/$DST_MODEL_NAME

python $ROOT/sparsification/obcq/obcq.py $SRC_MODEL $DATASET \
       --recipe $RECIPE_DIR/$RECIPE_NAME.yaml \
       --save 1 \
       --deploy-dir $DST_MODEL

