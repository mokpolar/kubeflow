FROM tensorflow/tensorflow:2.2.0-gpu
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

RUN pip install kfserving numpy
RUN pip install image
#RUN apt-get install libnvinfer.so.6 libnvinfer_plugin.so.6


ENV APP_HOME /app
WORKDIR $APP_HOME
ADD triton_mobilenet_transformer.py /app/

RUN wget https://github.com/NVIDIA/triton-inference-server/releases/download/v1.11.0/v1.11.0_ubuntu1604.clients.tar.gz && tar -xvzf v1.11.0_ubuntu1604.clients.tar.gz
RUN pip install python/tensorrtserver-1.11.0-py3-none-linux_x86_64.whl


ENTRYPOINT ["python", "triton_mobilenet_transformer.py"]