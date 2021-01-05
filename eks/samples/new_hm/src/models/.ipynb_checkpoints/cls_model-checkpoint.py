import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.applications import DenseNet121, MobileNet, MobileNetV2, VGG16
# from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
import ssl

def load_backbone(name, n_class=1, input_shape=(512,512,3), pretrained_path=None):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # get backbone cnn network
    if name=='densenet121':
        backbone = DenseNet121(include_top=False, input_shape=input_shape)
    elif name=='mobilenet-v1':
        backbone = MobileNet(include_top=False, input_shape=input_shape)
    elif name=='mobilenet-v2':
        backbone = MobileNetV2(include_top=False, input_shape=input_shape)
    elif name=='VGG16':
        backbone = VGG16(include_top=False, input_shape=input_shape)
    elif name=='efficientnet-b0':
        backbone = EfficientNetB0(include_top=False, input_shape=input_shape)
    elif name=='efficientnet-b1':
        backbone = EfficientNetB1(include_top=False, input_shape=input_shape)
    elif name=='efficientnet-b2':
        backbone = EfficientNetB2(include_top=False, input_shape=input_shape)
    elif name=='efficientnet-b3':
        backbone = EfficientNetB3(include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f'Not Supported backbone network = [{name}]')

    # set input layer
    inputs = L.Input(shape=input_shape)

    # get feature map [N, H/32, W/32, CH]
    feature_map = backbone(inputs)

    # get feature [N, CH]
    features = L.GlobalAveragePooling2D()(feature_map)

    # output - 뇌출혈유무 cls
    output = L.Dense(n_class)(features)
    output = L.Activation('sigmoid')(output)

    model = M.Model(inputs, [output, features, feature_map])

    # load pretrained weights
    if pretrained_path is not None:
        model.load_weights(pretrained_path)
        print(f'[Success] Pretrained weights was loaded. [{pretrained_path}]')

    return model


if __name__=='__main__':

    from utils.set_gpu import set_GPU
    set_GPU(-1)
    model = load_backbone('efficientnet-b0')
    model.summary()
