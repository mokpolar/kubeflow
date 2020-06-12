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

# for MobileNet Prediction Parsing
from tensorflow.keras.applications.mobilenet import decode_predictions
import tensorflow as tf


class KFServingSampleModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        

    def load(self):
        model_raw = tf.keras.experimental.load_from_saved_model('/app/saved_models/')
        
        # add model to class
        self._model = model_raw
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        # inputs : img file path ex) /tmp/img/cat.jpg
        inputs = request["instances"]


        # KFServing - Transformer
        # Preprocess
        img = Image.open(inputs[0])

        img = img.rotate(90).rotate(90).rotate(90).rotate(90)
        img = img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_LEFT_RIGHT)
        img = img.resize((224, 224))
        img = np.array(img)
        img = img/255
        img = img.reshape(-1, 224, 224, 3)

        # KFServing - Predictor
        logging.info('start predicting!')
        loaded_model = self._model
        yhat = loaded_model.predict(img)


        # KFServing - Transformer
        label = decode_predictions(yhat)
        label = label[0][0]
        results = [label[1], str(round(label[2]*100, 2))+'%']
        logging.info(results)


        return inputs[0]#{"predictions": inputs[0]}#{"predictions": results}


if __name__ == "__main__":
    model = KFServingSampleModel("custom-predictor")
    model.load()
    kfserving.KFServer(workers=1).start([model])