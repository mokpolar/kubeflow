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

from tensorflow.keras.applications.mobilenet import decode_predictions
import tensorflow as tf


logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)


def image_transform(instance):
    img = base64.b64decode(instance['image']['b64'])
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = img/255
    img = img.reshape(-1, 224, 224, 3)
    return img.tolist()

def parsing_prediction(prediction):
    label = decode_predictions(np.asarray([prediction]))
    label = label[0][0]
    output = [label[1], str(round(label[2]*100, 2))+'%']
    return output


class ImageTransformer(kfserving.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    def preprocess(self, inputs: Dict) -> Dict:
        return {'instances': image_transform(inputs['instances'][0])}

    def postprocess(self, inputs: Dict) -> Dict:
        start_post = time.time()
        return {'predictions': [parsing_prediction(prediction) for prediction in inputs['predictions']]}


if __name__ == "__main__":
    DEFAULT_MODEL_NAME = "mobilenet"

    parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                        help='The name that the model is served under.')
    parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

    args, _ = parser.parse_known_args()

    transformer = ImageTransformer(args.model_name, predictor_host=args.predictor_host)
    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])
