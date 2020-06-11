import kfserving
from typing import List, Dict
import logging
import boto3


# add
import base64
import io
import json
import time
import numpy as np
import argparse


# PIL
from PIL import Image
from PIL import ImageFilter


# for MobileNet Prediction Parsing
from tensorflow.keras.applications.mobilenet import decode_predictions
import tensorflow as tf

logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)

session = boto3.Session()
client = session.client('s3', endpoint_url='http://minio-service:9000', aws_access_key_id='minio', aws_secret_access_key='minio123')


# preprocess will decode b64 image, resize, normalize, reshaping

def image_transform(image):
    img = Image.open(image)
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
    print('image preprocessing completed')
    return img.tolist()

# postprocess will parse the prediction and get class name, accuracy

def parsing_prediction(prediction):
    # 여기서 멈추거나 
    print('decoding start!')
    label = decode_predictions(np.asarray([prediction]))
    print('decoding complete!')
    label = label[0][0]
    output = [label[1], str(round(label[2]*100, 2))+'%']
    print('complete parsing!')
    print('this is output!')
    return output



class ImageTransformer(kfserving.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self._key = None
    # Dict형 변수 input를 받아서 Dict 형 값을 반환. 
    def preprocess(self, inputs: Dict) -> Dict:
        bucket = inputs['Records'][0]['s3']['bucket']['name'] # bucket을 그때 지정. input의 Record에 s3 bucket 의 bucket anme을 갖고 와 -> 이건 event에 이렇게 들어 있겠지
        key = inputs['Records'][0]['s3']['object']['key'] # key는 Record에 key도 같이 들어있을테니까.. 파일 위치가 들어있지. 
        self._key = key
        client.download_file(bucket, key, '/tmp/' + key) # bucket지정한 데서 key를 갖고 와서 /tmp/ 폴더에 key로 일단 그 파일을  저장하고
        request = image_transform('/tmp/' + key)
        return {"instances": request} 
        
        
        '''
        if inputs['EventType'] == 's3:ObjectCreated:Put': # event type이 s3:ObjectCreate:Put 이면,
            bucket = inputs['Records'][0]['s3']['bucket']['name'] # bucket을 그때 지정. input의 Record에 s3 bucket 의 bucket anme을 갖고 와 -> 이건 event에 이렇게 들어 있겠지
            key = inputs['Records'][0]['s3']['object']['key'] # key는 Record에 key도 같이 들어있을테니까.. 파일 위치가 들어있지. 
            self._key = key
            print('bucket : ', bucket)
            print('key : ', key)
            client.download_file(bucket, key, '/tmp/' + key) # bucket지정한 데서 key를 갖고 와서 /tmp/ 폴더에 key로 일단 그 파일을  저장하고
            print('pre2!')
            # print(image_transform('/tmp/' + key))
            print('pre2.5!')
            request = image_transform('/tmp/' + key)
            print('this is request!')
            return {"instances": [request[0]]} # 내 transformer도 이렇게 하면 되겠네. 
        raise Exception("unknown event, pre3!")
        '''

    def postprocess(self, inputs: Dict) -> Dict:
        print('post1!')
        print('this is inputs!')
        print('post2!')
        #client.upload_file('/tmp/' + self._key, 'class', self._key)
        classified = [parsing_prediction(prediction) for prediction in inputs['predictions']]
        logging.info(classified)
        return inputs#{'predictions': [parsing_prediction(prediction) for prediction in inputs['predictions']]}



if __name__ == "__main__":
    DEFAULT_MODEL_NAME = "mnist"

    parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                        help='The name that the model is served under.')
    parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

    args, _ = parser.parse_known_args()

    transformer = ImageTransformer(args.model_name, predictor_host=args.predictor_host)
    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])