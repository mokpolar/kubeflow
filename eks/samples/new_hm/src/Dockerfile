FROM nvidia/cuda:10.1-base-ubuntu18.04
FROM tensorflow/tensorflow:2.1.0-gpu-py3


# Additional Packages
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0

# Added 20200909
RUN apt-get install -y ffmpeg
RUN pip install natsort


RUN pip install contextlib2 opencv-python

### for kfserving
RUN pip install kfserving==0.4.0
RUN pip install GPUtil

### for hm model
RUN pip install numpy
RUN pip install pandas
RUN pip install image
RUN pip install matplotlib

# secure FTP add
RUN pip install pysftp


ENV APP_HOME /app
WORKDIR $APP_HOME
ADD utils /app/utils
ADD fonts /app/fonts
ADD save_weights /app/save_weights
ADD save_pb /app/save_pb
ADD models /app/models
ADD data_load_functions.py /app/
ADD group_norm.py /app/
ADD hm_inference.py /app/


ENTRYPOINT ["python", "hm_inference.py"]