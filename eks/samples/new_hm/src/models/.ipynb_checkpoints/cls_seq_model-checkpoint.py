import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
from models.cls_model import load_backbone
from models.rnn_model import SequenceModel
# from utils.set_gpu import set_GPU
import numpy as np

def CLS_SEQ_MODEL(backbone, input_shape=(512,512,3), pretrained_path=None, backbone_pretrained_path=None):
    # get model
    cls_model = load_backbone(name=backbone, input_shape=input_shape, pretrained_path=backbone_pretrained_path)
    rnn_model = SequenceModel()

    inputs = L.Input(shape=input_shape)

    # cls in/out
    cls_outputs = cls_model(inputs)

    # [0]-->label output(N, 1), [1]-->features(N, CH), [2]-->feature map[N, H, W, CH]
    cls_output_label = cls_outputs[0]
    cls_output_feature = cls_outputs[1]
    cls_output_featuremap = cls_outputs[2]
    n_last_feature = cls_output_feature.shape[1]

    # reshape to be rnn inputs
    cls_output_label = K.reshape(cls_output_label,(1, -1, 1, 1))                   # [1, SeqLen, 1, 1]
    cls_output_feature = K.reshape(cls_output_feature, (1, -1, 1, n_last_feature)) # [1, SeqLen, 1, CH]

    # rnn in/out
    rnn_outputs = rnn_model([cls_output_feature, cls_output_label])

    rnn_output_label = rnn_outputs[1]
    rnn_output_patient = rnn_outputs[2]

    model = M.Model(inputs, [cls_output_label, rnn_output_label, rnn_output_patient, cls_output_featuremap])

    if pretrained_path is not None:
        model.load_weights(pretrained_path)
        print(f'[Success] Pretrained weights was loaded. [{pretrained_path}]')

    return model


# if __name__=='__main__':
#     # set cpu, gpu device
#     set_GPU(-1)

#     # Test
#     sample = np.zeros((30,256,256,3))
#     model = CLS_SEQ_MODEL(backbone='mobilenet-v1', input_shape=(256,256,3), pretrained_path=None)
#     outputs = model(sample)

#     model.summary()

#     print('CLS output label shape:', outputs[0].shape)
#     print('RNN output label shape:', outputs[1].shape)
#     print('Patient output shape:', outputs[2].shape)

