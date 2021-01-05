#!/bin/bash
StartTime=$(date +%s)

CLUSTER_IP=aa564448d92ed4a06b4a33035ab51471-1399444368.us-east-1.elb.amazonaws.com
SERVICE_HOSTNAME=mobilenet.kfserving-test.example.com
INPUT_PATH=@./cat.json
MODEL_NAME=my-model

i=1
while [ $i -lt 30 ]

do 
  curl -v -H "Host: ${SERVICE_HOSTNAME}" http://$CLUSTER_IP/v1/models/$MODEL_NAME:predict -d $INPUT_PATH
  i=$(($i+1))
done

EndTime=$(date +%s)
echo "\n"
echo "Inference takes $(($EndTime - $StartTime)) seconds"
