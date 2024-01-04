#!/bin/bash

ROOT=$HOME/work/llama2.cnn_dailymail.train/src/neuralmagic/sparseml
source $ROOT/my_scripts/start_here.sh

python $ROOT/my_scripts/test.py
