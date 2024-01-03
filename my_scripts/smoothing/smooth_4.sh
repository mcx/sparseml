#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

SRC_ROOT=$HOME/work/llama2_triviaqa_sparsegpt/src/neuralmagic/sparseml/src/sparseml/transformers/sparsification/obcq
RECIPE_DIR=$HOME/work/llama2_triviaqa_sparsegpt/src/neuralmagic/sparseml/my_recipes/smoothing

MODEL_DIR=/data/tuan/models/llama/TriviaQA/rc.wikipedia.nocontext

#SRC_MODEL_DIR=$MODEL_DIR/pruned_finetuned
#SRC_ID=27967
#SRC_MODEL=$SRC_MODEL_DIR/sparse_ft@SRC21550@lr8e-5@WD0.001@B32@GrAcc2@W0.1@ep5@GPUs8@ID$SRC_ID/hf

#SRC_MODEL_DIR=$MODEL_DIR/pruned_finetuned
#SRC_ID=32177
#SRC_MODEL=$SRC_MODEL_DIR/sparse_ft@SRC21155@lr8e-5@WD0.001@B32@GrAcc1@W0.1@ep2@GPUs8@ID$SRC_ID/hf

SRC_MODEL_DIR=$MODEL_DIR
SRC_ID=28459
SRC_MODEL=$SRC_MODEL_DIR/Llama-2-7b-hf@trivia_qa@lr3e-5@B64@GrAcc1@W0.1@ep2@GPUs4@WD0.0@ID$SRC_ID


DST_MODEL_DIR=$MODEL_DIR/smoothed

NSAMPLES=1024
for RECIPE_NAME in smoothquant05 smoothquant06 smoothquant07 smoothquant08 smoothquant09
do

ID=$RANDOM
DST_MODEL=$DST_MODEL_DIR/sparsegpt@SRC$SRC_ID@$RECIPE_NAME@N$NSAMPLES@ID$ID

python $SRC_ROOT/obcq.py $SRC_MODEL trivia_qa \
       --recipe $RECIPE_DIR/$RECIPE_NAME.yaml \
       --nsamples $NSAMPLES \
       --save 1 \
       --deploy-dir $DST_MODEL

done
