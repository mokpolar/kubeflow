FROM tensorflow/tensorflow:2.0.0-py3

RUN pip install kfserving==0.3.0 numpy
RUN pip install image

ENV APP_HOME /app
WORKDIR $APP_HOME
ADD cat.jpg /tmp/img/
ADD saved_models /app/saved_models
ADD custom_predictor.py /app/

CMD ["python", "custom_predictor.py"]