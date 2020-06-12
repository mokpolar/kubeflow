FROM python:3.6-slim

RUN pip install torch torchvision
RUn pip install kfserving==0.3.0

ENV APP_HOME /app
WORKDIR $APP_HOME
ADD model.py /app/
ADD model.pt /app/

CMD ["python", "custom_model.py"]