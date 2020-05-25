import argparse
import base64
import io
import logging
from typing import Dict

import json

# pip install kfserving==0.3.0 -> dockerfile
import kfserving
# pip install numpy -> dockerfile
import numpy as np

# pip install opencv-python -> dockerfile
from PIL import Image
from PIL import ImageFilter

import time

start = time.time()

# for MobileNet Prediction Parsing
from tensorflow.keras.applications.mobilenet import decode_predictions
import tensorflow as tf


logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)


# preprocess will decode b64 image, resize, normalize, reshaping

def image_transform(instance):
    img = base64.b64decode(instance['image']['b64'])
    img = Image.open(io.BytesIO(img))
    # complexing code
    img = img.rotate(90).rotate(90).rotate(90).rotate(90)
    img = img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_TOP_BOTTOM)
    img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_LEFT_RIGHT)
    # 블러처리
    img = img.filter(ImageFilter.BLUR)
    # 엠보싱
    img = img.filter(ImageFilter.EMBOSS)
    # 윤곽선 변환
    img = img.filter(ImageFilter.CONTOUR)
    # 자세히
    img = img.filter(ImageFilter.DETAIL)
    # 날카롭게
    img = img.filter(ImageFilter.EDGE_ENHANCE)
    # 부드럽게
    img = img.filter(ImageFilter.SMOOTH)

    img = img.resize((224, 224))
    img = np.array(img)
    img = img/255
    img = img.reshape(-1, 224, 224, 3)
    logging.info(img)
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

    def preprocess(self, inputs: Dict) -> Dict:
        return {'instances': image_transform(inputs['instances'][0])}

    def postprocess(self, inputs: Dict) -> Dict:
        end = time.time()
        return {'predictions': [parsing_prediction(prediction) for prediction in inputs['predictions']], 'time' : [end - start]}


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