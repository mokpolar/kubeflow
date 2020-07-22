#!/bin/bash
StartTime=$(date +%s)

CLUSTER_IP=a928c7ef9e005401fb7c58d9a6f71d73-1684550075.us-east-1.elb.amazonaws.com
SERVICE_HOSTNAME=hm-model.default.example.com
INPUT_PATH=@./input.json
MODEL_NAME=hm-model
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://$CLUSTER_IP/v1/models/$MODEL_NAME:predict -d $INPUT_PATH

EndTime=$(date +%s)
echo "\n"
echo "Inference takes $(($EndTime - $StartTime)) seconds"

