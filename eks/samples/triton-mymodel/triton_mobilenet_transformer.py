from tensorflow.python.client import device_lib
import argparse
import base64
import io
import logging
from typing import Dict

import json

import kfserving
import numpy as np

from PIL import Image

import time

# for MobileNet Prediction Parsing
from tensorflow.keras.applications.mobilenet import decode_predictions
import tensorflow as tf

# for NVIDIA Triton Server
from tensorrtserver.api import *

logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)


# preprocess will decode b64 image, resize, normalize, reshaping

def image_transform(instance):
    img = base64.b64decode(instance['image']['b64'])
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = img/255
    #img = img.reshape(-1, 224, 224, 3)
    logging.info("img reshaping complete. {}".format(time.time()))
    return img.tolist()

# postprocess will parse the prediction and get class name, accuracy

def parsing_prediction(prediction):
    label = decode_predictions(np.asarray([prediction]))
    label = label[0][0]
    output = [label[1], str(round(label[2]*100, 2))+'%']
    return output


class ImageTransformer(kfserving.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.model_name = "my_model"
        self.model_version = -1
        self.protocol = ProtocolType.from_str('http')

        self.infer_ctx = None

    def preprocess(self, inputs: Dict) -> Dict:
        logging.info("start preprocessing. {}".format(time.time())) # when input arrives
        self.inputs = image_transform(inputs['instances'][0])
        return self.inputs


    def predict(self, inputs: Dict) -> Dict:
        logging.info("start predicting. {}".format(time.time()))
        logging.info(type(inputs))
        inputs = np.array(inputs, dtype = np.float32)

        if not self.infer_ctx:

            self.infer_ctx = InferContext(self.predictor_host, self.protocol, self.model_name, self.model_version, http_headers='', verbose=True)

        result = self.infer_ctx.run({'input_1': [inputs]},{'act_softmax' : InferContext.ResultFormat.RAW})

        return result  

    def postprocess(self, result: Dict) -> Dict:
        logging.info("start postprocessing. {}".format(time.time())) # when prediction arrvies
        logging.info(device_lib.list_local_devices())
        return {'predictions': [parsing_prediction(prediction) for prediction in result['act_softmax']]}


if __name__ == "__main__":
    DEFAULT_MODEL_NAME = "keras-mobilenet"

    parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                        help='The name that the model is served under.')
    parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

    args, _ = parser.parse_known_args()

    transformer = ImageTransformer(args.model_name, predictor_host=args.predictor_host)
    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])