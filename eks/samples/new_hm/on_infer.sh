#!/bin/bash
StartTime=$(date +%s%N)


CLUSTER_IP=a7b0fc16c3bc54a278ac63ba9ee837a9-688394741.us-east-1.elb.amazonaws.com
SERVICE_HOSTNAME=hm-model.kfserving-test.example.com
INPUT_PATH=@./on_input.json
MODEL_NAME=hm-model
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://$CLUSTER_IP/v1/models/$MODEL_NAME:predict -d $INPUT_PATH

EndTime=$(date +%s%N)

