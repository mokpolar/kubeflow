FROM python:3.6-slim

RUN pip install torch torchvision
RUN pip install kfserving==0.3.0 image numpy

ENV APP_HOME /app
WORKDIR $APP_HOME
ADD image_transformer.py /app/

ENTRYPOINT ["python", "image_transformer.py"]
