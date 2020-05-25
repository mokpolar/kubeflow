CLUSTER_IP=abcf45710b2ff4858b1b65c5d69b1bd9-1732374792.us-east-1.elb.amazonaws.com
SERVICE_HOSTNAME=complex-mobilenet-transformer.mobilenet.example.com
INPUT_PATH=@./cat.json
MODEL_NAME=complex-mobilenet-transformer
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://$CLUSTER_IP/v1/models/$MODEL_NAME:predict -d $INPUT_PATH
