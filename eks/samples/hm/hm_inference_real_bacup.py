'''

뇌출혈 과제 inference 과정 샘플 코드

tensorflow 2버전
python 3.6
opencv 

'''
from typing import Dict

import json
import logging

import kfserving

import cv2
import numpy as np
import pandas as pd
import os, sys

import data_load_functions as dl
import models.model_definition as md

from utils.save_img import save_img
from utils.get_binary_CAM import get_binary_CAM


# jy customizing
import tensorflow as tf



def set_gpu(gpu_id='AUTO'):
    '''  gpu_id : 'AUTO' : 자동  '1' : 1번 GPU  'CPU' : CPU   '''
    if gpu_id == 'AUTO':       gpu_id = str(GPUtil.getAvailable()[0])
    if gpu_id == 'CPU' :       gpu_id = '-1'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return gpu_id



set_gpu('CPU')
print('set CPU')
logging.info('set CPu')        

# ====================================== load data
# 딱 한명분 로드 함
PATIENT_ID = 0
images = dl.get_A_subject(df_path='./dataset/train.csv', img_dir='./dataset/imgs', patient_id=PATIENT_ID)
# images - (36, 512, 512, 3)



# ======================================= prepare tensorflow and keras
# 텐서플로 준비하는 코드를 삽입합니다. 
# 모델 준비, 모델 로드 포함
# 모델 정의는 define_models.py 에.
model_ano = md.define_model_ano()
#model_ano.load_weight("/Users/mokpolar/Documents/GitHub/kubeflow/eks/samples/hm/weight/seg/")
#model_ano = tf.keras.models.load_model("./weight/seg/")
logging.info('model ano loaded')
#model_ano.summary()



model_cls = md.define_model_cls()
#model_cls.load_weights("/Users/mokpolar/Documents/GitHub/kubeflow/eks/samples/hm/weight/cls/")
#model_cls = tf.keras.models.load_model("./weight/cls/")
logging.info('model cls loaded')

#model_cls.summary()

model_rnn = md.define_model_rnn()
#model_rnn.load_weights("/Users/mokpolar/Documents/GitHub/kubeflow/eks/samples/hm/weight/rnn/")
#model_rnn = tf.keras.models.load_model("/Users/mokpolar/Documents/GitHub/kubeflow/eks/samples/hm/weight/rnn")
#model_rnn = tf.saved_model.load("/Users/mokpolar/Documents/GitHub/kubeflow/eks/samples/hm/weight/rnn")

#model_rnn.summary()



# ======================================== pre processing
# 전처리 수행
# 히스토그램 평활화를 수행
# 추가적인 프로세스 있을수 있음
images_new = []
for image in images:
#     image = np.uint8(np.clip(np.squeeze(image*255),0,255))
    image = np.uint8(np.clip(np.squeeze(image),0,255))
    image = np.stack( [cv2.equalizeHist(image[:,:,c]) for c in range(3)], -1)
    image = np.float32(image/255)
    images_new.append( image)         
images = np.stack(images_new) # [N, H, W, 3]
print('Preprocessing 완료...')
    
    

    
# ========================================= model 1
# 첫번째 모델 동작
# 여기서는 Anomaly detection
preds_ano = []
for i in range(images.shape[0]):
    image = images[i:i+1,:,:,:]
    image = image[:,:,:,1:2] # 임시로 1채널만 작동하게
    pred = model_ano(image)
    pred = np.array(pred)
    preds_ano.append(pred)

preds_ano = np.concatenate(preds_ano) # [N, H, W, 1]
print('Ano 완료...')


# ========================================= model 2
# 두번째 모델 동작
# 여기서는 classification
# model_cls 의 입력은 [512,512,3+1] (CT영상 3장 + 앞단의 결과(Seg혹은 AnoDet))

inputs_cls = np.concatenate([images, preds_ano], axis=-1) # [N, H, W, 4]

preds_cls = []
features_cls = []
cams_cls = []
for i in range(inputs_cls.shape[0]):
    prediction_cls, fea_cls, heatmap_CAM = model_cls(np.expand_dims(inputs_cls[i], axis=0))
    preds_cls.append(prediction_cls)
    cams_cls.append(heatmap_CAM)
    features_cls.append(fea_cls)
    
preds_cls = np.array(preds_cls) # [LenSeq, 1, 1]
features_cls = np.array(features_cls) # [LenSeq, 1, 1024]
cams_cls = np.concatenate(cams_cls, axis=0) # [LenSeq, 16, 16, 1024]
print('Classification 완료...')



# ========================================= model 3
# 세번째 모델 동작.
# 여기서는 sequntial network(RNN)

preds_cls = np.expand_dims(preds_cls, axis=0) # [1, LenSeq, 1, 1]
features_cls = np.expand_dims(features_cls, axis=0) # [1, LenSeq, 1, LenFeat]

_, pred_rnn = model_rnn((features_cls, preds_cls))
pred_rnn = np.reshape(pred_rnn, (-1, 1)) # [LenSeq, 1]
print('RNN 완료...')




# ========================================= Postprocessing
# 후처리
# cls CAM 처리
    
# get last fc weight
weights = model_cls.layers[-1].get_weights()[0]

# get cam, overlay imgs
cam_imgs = []
overlay_imgs = []
for i, cam_feature_map in enumerate(cams_cls):
    
    cam_img, overlay_img = get_binary_CAM(images[i], cam_feature_map, weights)
    cam_imgs.append(cam_img)
    overlay_imgs.append(overlay_img)
    
# list to numpy
cam_imgs = np.array(cam_imgs)
overlay_imgs = np.array(overlay_imgs)




# ========================================== save
# 결과 저장하는 코드
# seg 결과, cls CAM 결과, rnn slice 결과, rnn 환자 단위 결과


# 저장할 환자 단위 폴더 생성
SAVE_DIR = f'./outputs/{PATIENT_ID:03d}_Patient'
os.makedirs(SAVE_DIR, exist_ok=True)

# Seg 결과 저장
THRESHOLD_ANO = 0.5
for i, pred_ano in enumerate(preds_ano):
    # thresholding
    img = pred_ano>THRESHOLD_ANO
    # save thresholded imgs
    save_img(os.path.join(SAVE_DIR, f'{i:02d}_slice.png'), img, verbose=0)
    # save confidence imgs
    save_img(os.path.join(SAVE_DIR, f'{i:02d}_slice_confidence.png'), pred_ano, verbose=0)
print('Seg 결과 저장 완료...')
    
# Cls CAM 결과 저장
for i, cam_img in enumerate(overlay_imgs):
    # save imgs
    save_img(os.path.join(SAVE_DIR, f'{i:02d}_slice_cam.png'), cam_img, verbose=0)
print('CAM 결과 저장 완료...')


# rnn slice별 및 환자의 뇌출혈 유뮤 결과 저장
# thresholding
THRESHOLD_RNN = 0.5
pred_hm_rnn = pred_rnn>THRESHOLD_RNN

tmp_pred = np.concatenate([pred_rnn, pred_hm_rnn], axis=-1)

# np -> dataframe
df = pd.DataFrame(tmp_pred, columns=['Score', 'Hemorrhage'])

# save
df.to_csv(os.path.join(SAVE_DIR, 'slice_hm_score.csv'))
print('Slice hemorrhage score 저장완료...')
    