#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

ROOT=$HOME/src/neuralmagic/sparseml/src/sparseml/experimental/sparsegpt

DATASET=gsm8k

SRC_MODEL_ORG_OR_DIR=/data/models/tuan/llama
SRC_MODEL_NAME=Llama-2-7b-hf@gsm8k@lr3e-5@B16@W0.1@ep2@GPUs4
SRC_MODEL=$SRC_MODEL_ORG_OR_DIR/$SRC_MODEL_NAME

SP=0.6

ID=$RANDOM

DST_MODEL_DIR=/data/models/tuan/llama/ongoing
DST_MODEL_NAME=sparsegpt@@@$SRC_MODEL_NAME@@@$DATASET@SP$SP@ID$ID
DST_MODEL=$DST_MODEL_DIR/$DST_MODEL_NAME

OBSERVER_BATCHES=128

python $ROOT/main.py $SRC_MODEL $DATASET \
       --data-sequence-length 1024 \
       --sparsity $SP \
       --observer-batches $OBSERVER_BATCHES \
       --save $DST_MODEL

cp "$0" $DST_MODEL/command.sh
