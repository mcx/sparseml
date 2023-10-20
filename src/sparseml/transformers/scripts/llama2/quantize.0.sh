#!/bin/bash

ROOT=$HOME/src/neuralmagic/sparseml/src/sparseml/transformers

RECIPE_DIR=$ROOT/recipes/llama2
RECIPE_NAME=example_llama

SRC_MODEL_DIR=/network/tuan/models/llama/GSM8K/base_finetuned_pruned
SRC_MODEL_NAME=llama-2-7b_pruned60
SRC_MODEL=$SRC_MODEL_DIR/$SRC_MODEL_NAME

DATASET=gsm8k

python $ROOT/sparsification/obcq/obcq.py $SRC_MODEL $DATASET \
       --recipe $RECIPE_DIR/$RECIPE_NAME.yaml

