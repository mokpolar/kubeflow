FROM tensorflow/tensorflow:2.1.0-py3


RUN pip install kfserving==0.3.0 numpy image


ENV APP_HOME /app
WORKDIR $APP_HOME
ADD image_transformer.py /app/

ENTRYPOINT ["python", "image_transformer.py"]
