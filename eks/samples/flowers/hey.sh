MODEL_NAME=flowers-sample
INPUT_PATH=./input.json
HOST=flowers-sample.kfserving-test.example.com
hey -z 30s -c 100 -m POST -host ${HOST} -D ${INPUT_PATH} http://a7b0fc16c3bc54a278ac63ba9ee837a9-688394741.us-east-1.elb.amazonaws.com/v1/models/$MODEL_NAME:predict
