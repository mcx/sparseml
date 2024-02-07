#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SRC_ROOT=$HOME/work/llama2.cnn_dailymail.sparsegpt/src/neuralmagic/sparseml/src/sparseml/transformers/sparsification/obcq
RECIPE_DIR=$HOME/work/llama2.cnn_dailymail.sparsegpt/src/neuralmagic/sparseml/my_recipes

MODEL_DIR=$HOME/models/llama2/cnn_dailymail

SRC_ID=28459
SRC_MODEL_DIR=$MODEL_DIR/llama-recipes/dense_finetuned
SRC_MODEL=$SRC_MODEL_DIR/Llama-2-7b-hf@trivia_qa@lr3e-5@B64@GrAcc1@W0.1@ep2@GPUs4@WD0.0@ID$SRC_ID

SRC_ID=2967
SRC_MODEL=$HOME/models/llama2/cnn_dailymail/llama-recipes/dense_finetuned/Llama-2-7b-hf@lr8e-5@B16@GrAcc4@W0.1@ep1@GPUs8@WD0.0@ID$SRC_ID

DST_MODEL_DIR=$MODEL_DIR/owl

SEQ_LEN=2048

SP=40

for M in 5
do
for LMBDA in 0.02
do
for NSAMPLES in 16
do

ID=$RANDOM
RECIPE_NAME=prune$SP.owl-lmbda$LMBDA-m$M.test
DST_MODEL=$MODEL_DIR/sparsegpt@SRC$SRC_ID@$RECIPE_NAME@N$NSAMPLES@ID$ID


python $SRC_ROOT/obcq.py $SRC_MODEL cnn-dailymail \
       --recipe $RECIPE_DIR/$RECIPE_NAME.yaml \
       --nsamples $NSAMPLES \
       --seqlen $SEQ_LEN \
       --save 0 \
       --eval cnn-dailymail \
       --deploy-dir $DST_MODEL

done
done
done
