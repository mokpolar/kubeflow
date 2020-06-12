#!/bin/bash
StartTime=$(date +%s)

CLUSTER_IP=a0b2f82d28e1e4f93ab7cab22a34bf49-1330092172.us-east-1.elb.amazonaws.com
SERVICE_HOSTNAME=custom-predictor.default.example.com
INPUT_PATH=@./cat.jpg
MODEL_NAME=custom-predictor
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://$CLUSTER_IP/v1/models/$MODEL_NAME:predict -d $INPUT_PATH

EndTime=$(date +%s)
echo "\n"
echo "Inference takes $(($EndTime - $StartTime)) seconds"

