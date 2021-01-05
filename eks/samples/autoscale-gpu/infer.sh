#!/bin/bash
StartTime=$(date +%s)

CLUSTER_IP=a8ee37c407eac4fe59a7822207bce826-1938513030.us-east-1.elb.amazonaws.com
MODEL_NAME=flowers-sample-gpu
INPUT_PATH=./input.json
HOST=$(kubectl -n kfserving-test get inferenceservice $MODEL_NAME -o jsonpath='{.status.url}' | cut -d "/" -f 3)
hey -z 30s -c 5 -m POST -host ${HOST} -D $INPUT_PATH http://$CLUSTER_IP/v1/models/$MODEL_NAME:predict

EndTime=$(date +%s)
echo "\n"
echo "Inference takes $(($EndTime - $StartTime)) seconds"

