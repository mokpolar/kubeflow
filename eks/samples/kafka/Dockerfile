FROM tensorflow/tensorflow:2.1.0-py3


RUN pip install kfserving==0.3.0 numpy
RUN pip install image
RUN pip install boto3


ENV APP_HOME /app
WORKDIR $APP_HOME
RUN mkdir ~/.keras
RUN mkdir ~/.keras/models
RUN curl -o ~/.keras/models/imagenet_class_index.json -L https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
ADD kafka_image_transformer.py /app/

ENTRYPOINT ["python", "kafka_image_transformer.py"]