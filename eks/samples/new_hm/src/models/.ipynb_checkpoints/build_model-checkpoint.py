from models.cls_model import load_backbone
from models.rnn_model import SequenceModel
from models.cls_seq_model import CLS_SEQ_MODEL


def build_model(model_type, backbone='densenet121', input_shape=(512, 512, 3), pretrained_path=None, backbone_pretrained_path=None):
    """model type에 따라 모델 반환, [cls, cls_seq]의 경우는 backbone에 따라 모델 생성"""
    if model_type == 'cls':
        return load_backbone(name=backbone, input_shape=input_shape, pretrained_path=pretrained_path)
    elif model_type == 'rnn':
        return SequenceModel()
    elif model_type == 'cls_seq':
        return CLS_SEQ_MODEL(backbone=backbone, input_shape=input_shape, pretrained_path=pretrained_path, backbone_pretrained_path=backbone_pretrained_path)

    else:
        raise ValueError(f'Not Supported type==[{model_type}]')


# if __name__ == '__main__':
#     from utils.set_gpu import set_GPU

#     set_GPU(-1)
#     model = build_model('mobilenet-v1')
#     model.summary()