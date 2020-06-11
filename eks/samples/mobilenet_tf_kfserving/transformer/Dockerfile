FROM tensorflow/tensorflow:2.1.0-py3


RUN pip install kfserving==0.3.0 numpy
RUN pip install image


ENV APP_HOME /app
WORKDIR $APP_HOME
ADD simple_mobilenet_transformer.py /app/

ENTRYPOINT ["python", "simple_mobilenet_transformer.py"]