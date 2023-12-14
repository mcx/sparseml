#!/bin/bash

CUDA_VISIBLE_DEVICES=1

SRC_ROOT=$HOME/work/llama2_triviaqa_sparsegpt/src/neuralmagic/sparseml/src/sparseml/transformers/sparsification/obcq
RECIPE_DIR=$HOME/work/llama2_triviaqa_sparsegpt/src/neuralmagic/sparseml/my_recipes/quantize

MODEL_DIR=/data/tuan/models/llama/TriviaQA/rc.wikipedia.nocontext/pruned_finetuned

SRC_ID=27967
SRC_MODEL=$MODEL_DIR/sparse_ft@SRC21550@lr8e-5@WD0.001@B32@GrAcc2@W0.1@ep5@GPUs8@ID$SRC_ID/hf

DST_MODEL_DIR=$MODEL_DIR/quantized

NSAMPLES=2048

for RECIPE_NAME in dense_w8_loge_smooth.v02 dense_w8a8_loge_smooth
do

ID=$RANDOM
DST_MODEL=$MODEL_DIR/sparsegpt@SRC$SRC_ID@$RECIPE_NAME@N$NSAMPLES@ID$ID

python $SRC_ROOT/obcq.py $SRC_MODEL trivia_qa \
       --recipe $RECIPE_DIR/$RECIPE_NAME.yaml \
       --nsamples $NSAMPLES \
       --save 1 \
       --deploy-dir $DST_MODEL

done
