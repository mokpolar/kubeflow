#!/bin/bash
StartTime=$(date +%s)

CLUSTER_IP=af56ac8f83bc04435ba6f540df40803b-2102140833.us-east-1.elb.amazonaws.com
SERVICE_HOSTNAME=custom-predictor.default.example.com
INPUT_PATH=@./input.json
MODEL_NAME=custom-predictor
SESSION=MTU5ODk2MDMwMnxOd3dBTkZZeVNsaEVURlJUUTFoTU5WaFBRVE5HVWxrMVNrNUhSRFpLTlZoWlVUUkRSRUZLTkZoYVNrTTJSRFkyVURKV1JqUk1URUU9fHitYKiqh-AYtuuNWqr-InZfZ5XXNUxywhRBLLeJRxTL
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Cookie: authservice_session=${SESSION}" http://$CLUSTER_IP/v1/models/$MODEL_NAME:predict -d $INPUT_PATH

EndTime=$(date +%s)
echo "\n"
echo "Inference takes $(($EndTime - $StartTime)) seconds"

