CLUSTER_IP=a52205f0c99a14d128958f8371128b64-765876472.us-east-1.elb.amazonaws.com
SERVICE_HOSTNAME=flowers-sample.kfserving-test.example.com
INPUT_PATH=@./input.json
MODEL_NAME=flowers-sample
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://$CLUSTER_IP/v1/models/$MODEL_NAME:predict -d $INPUT_PATH

i=1
while [ $i -lt 30 ]

do
  curl -v -H "Host: ${SERVICE_HOSTNAME}" http://$CLUSTER_IP/v1/models/$MODEL_NAME:predict -d $INPUT_PATH
  i=$(($i+1))
done
