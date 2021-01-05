#!/bin/bash
StartTime=$(date +%s)

CLUSTER_IP=a7b0fc16c3bc54a278ac63ba9ee837a9-688394741.us-east-1.elb.amazonaws.com
MODEL_NAME=flowers
INPUT_PATH=@./flower_input.json
SERVICE_HOSTNAME=flowers.kfserving-test.example.com
curl -v -H "Host: ${SERVICE_HOSTNAME}" -X POST http://$CLUSTER_IP/v2/models/$MODEL_NAME
echo "---"
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://$CLUSTER_IP/v1/models/$MODEL_NAME:predict -d $INPUT_PATH


EndTime=$(date +%s)
echo "\n"
echo "Inference takes $(($EndTime - $StartTime)) seconds"

