import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
from tensorflow.keras.applications import DenseNet121


def DenseNet(n_class=1, input_shape=(512,512,4)):
    
    densenet121 = DenseNet121(include_top=False, input_shape=(512,512,3))

    dense_input = L.Input(shape=input_shape)
    dense_filter = L.Conv2D(3, 3, padding='same')(dense_input)
    dense_output = densenet121(dense_filter)

    features = L.GlobalAveragePooling2D()(dense_output)
    output = L.Dense(n_class)(features)

    model = M.Model(dense_input, [output, features, dense_output])

    return model
    
    

    
    
    
    
    