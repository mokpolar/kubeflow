FROM python:3.6-slim


RUN pip install 
RUN pip install kfserving==0.3.0 image numpy



ENV APP_HOME /app
WORKDIR $APP_HOME
ADD mobilenet_transformer.py /app/

ENTRYPOINT ["python", "mobilenet_transformer.py"]