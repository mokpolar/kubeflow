#!/bin/bash
StartTime=$(date +%s)

CLUSTER_IP=a8ee37c407eac4fe59a7822207bce826-1938513030.us-east-1.elb.amazonaws.com
SERVICE_HOSTNAME=custom-predictor.kfserving-test.example.com
INPUT_PATH=@./input.json
MODEL_NAME=custom-predictor
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://$CLUSTER_IP/v1/models/$MODEL_NAME:predict -d $INPUT_PATH

EndTime=$(date +%s)
echo "\n"
echo "Inference takes $(($EndTime - $StartTime)) seconds"

