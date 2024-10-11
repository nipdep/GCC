#!/bin/bash
hidden_size=$1
model=$2
ARGS=${@:3}

for dataset in $ARGS
do
    python downstream_node_classification.py --dataset $dataset --hidden-size $hidden_size --model $model
done
