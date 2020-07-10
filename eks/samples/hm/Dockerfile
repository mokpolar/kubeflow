FROM tensorflow/tensorflow:2.1.0-py3

# Additional Packages
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0
RUN pip install contextlib2 opencv-python

### for kfserving
RUN pip install kfserving==0.3.0

### for hm model
RUN pip install numpy
RUN pip install pandas
RUN pip install image
RUN pip install matplotlib


ENV APP_HOME /app
WORKDIR $APP_HOME
ADD utils /app/utils
ADD weight /app/weight
ADD dataset /app/dataset
ADD models /app/models
ADD data_load_functions.py /app/
ADD group_norm.py /app/
ADD hm_inference.py /app/


ENTRYPOINT ["python", "hm_inference.py"]