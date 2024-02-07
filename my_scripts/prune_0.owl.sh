#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SRC_ROOT=$HOME/work/llama2.cnn_dailymail.sparsegpt/src/neuralmagic/sparseml/src/sparseml/transformers/sparsification/obcq
RECIPE_DIR=$HOME/work/llama2.cnn_dailymail.sparsegpt/src/neuralmagic/sparseml/my_recipes

MODEL_DIR=$HOME/models/llama2/cnn_dailymail

# Pretrained finetuned model
#SRC_ID=2967
#SRC_MODEL=$MODEL_DIR/llama-recipes/dense_finetuned/Llama-2-7b-hf@lr8e-5@B16@GrAcc4@W0.1@ep1@GPUs8@WD0.0@ID$SRC_ID

# 50% sparse finetuned model
SRC_ID=3641
SRC_MODEL=$MODEL_DIR/llama-recipes/sparse_finetuned/sparse_ft@SRC5043@lr8e-5@WD0.0@B8@GrAcc8@W0.1@ep1@GPUs7@ID$SRC_ID/hf

DST_MODEL_DIR=$MODEL_DIR/owl

M=5

for SP in 60
do
for LMBDA in 0.02
do
for NSAMPLES in 1024
do

ID=$RANDOM
RECIPE_NAME=prune$SP.owl-lmbda$LMBDA-m$M
DST_MODEL=$DST_MODEL_DIR/sparsegpt@SRC$SRC_ID@$RECIPE_NAME@N$NSAMPLES@ID$ID


python $SRC_ROOT/obcq.py $SRC_MODEL cnn-dailymail \
       --recipe $RECIPE_DIR/$RECIPE_NAME.yaml \
       --nsamples $NSAMPLES \
       --seqlen 2048 \
       --save 1 \
       --eval cnn-dailymail \
       --deploy-dir $DST_MODEL

done
done
done
