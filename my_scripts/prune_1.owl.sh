#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

SRC_ROOT=$HOME/work/llama2_triviaqa_sparsegpt/src/neuralmagic/sparseml/src/sparseml/transformers/sparsification/obcq
RECIPE_DIR=$HOME/work/llama2_triviaqa_sparsegpt/src/neuralmagic/sparseml/my_recipes

MODEL_DIR=/data/tuan/models/llama/TriviaQA/rc.wikipedia.nocontext

SRC_ID=28459
SRC_MODEL=$MODEL_DIR/Llama-2-7b-hf@trivia_qa@lr3e-5@B64@GrAcc1@W0.1@ep2@GPUs4@WD0.0@ID$SRC_ID

DST_MODEL_DIR=$MODEL_DIR/pruned

SP=60
for M in 5 7
do
for LMBDA in 0.02 0.05 0.08 0.1 0.2
do
for NSAMPLES in 1024
do

ID=$RANDOM
RECIPE_NAME=prune$SP.owl-lmbda$LMBDA-m$M
DST_MODEL=$MODEL_DIR/sparsegpt@SRC$SRC_ID@$RECIPE_NAME@N$NSAMPLES@ID$ID


python $SRC_ROOT/obcq.py $SRC_MODEL trivia_qa \
       --recipe $RECIPE_DIR/$RECIPE_NAME.yaml \
       --nsamples $NSAMPLES \
       --save 1 \
       --eval 1 \
       --deploy-dir $DST_MODEL
done
done
done
