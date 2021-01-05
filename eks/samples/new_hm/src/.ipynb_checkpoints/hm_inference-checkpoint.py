'''

뇌출혈 과제 inference 과정 샘플 코드

tensorflow 2버전
python 3.6
opencv 


hm : hemorrhage


TODO 
O 결과 score는  1~0 확률값으로 
O 40 256 윈도 데이터 로드하게 변경
O CNNRNN네트워크 저장 파일 변경
O cnn cls rnn seq pred, logit 용어 정리.
    cls seq 으로 표기
    pred : 모델 아웃풋.
    score : 0~1 확률값
    hmpred : 0,1 헤모 여부
- 결과 JSON 출력

'''



#====================================== SETTING

DEBUG = False                # 
SAVE_PB = False
SAVE_INTERMEDIATE = False

GPU_ID = 'CPU' # '0', 'AUTO', 'CPU'

PATH_MODEL_ANO = '../save_weights/ano/epoch-490'
PATH_MODEL_CLS_SEQ = '../save_weights/cls_seq/epoch_040'

DIR_INPUT_DATA = '../inputdata/HA000010'
DIR_OUTPUT = f'../outputs/'

### 작동 테스트용
# DIR_INPUT_DATA = '/mnt/g/Hemorrhage/00_AJOU/data_png/Hemo/HAJ________000008/'
# DIR_INPUT_DATA = '/mnt/g/Hemorrhage/00_AJOU/data_png/Normal/NAJ________000010/'
# DIR_INPUT_DATA = '/mnt/g/Hemorrhage/00_SNU/data_png/Hemo/HSN________CA1411/'
# DIR_INPUT_DATA = '/mnt/g/Hemorrhage/00_SNU/data_png_v02/Normal/NSN________000001/'
# DIR_INPUT_DATA = '/mnt/g/Hemorrhage/00_Kaggle_RSNA/data_png/Hemo/HKG________000050/'
# DIR_INPUT_DATA = '/mnt/g/Hemorrhage/00_Kaggle_RSNA/data_png/Hemo/HKG________000010/'
# DIR_INPUT_DATA = '/mnt/g/Hemorrhage/00_Kaggle_RSNA/data_png/Hemo/HKG________005010/'
# DIR_INPUT_DATA = '/mnt/g/Hemorrhage/00_Kaggle_RSNA/data_png/Normal/NKG________000570/'
# DIR_INPUT_DATA = '/mnt/g/Hemorrhage/00_Kaggle_RSNA/data_png/Normal/NKG________000570/'
# DIR_INPUT_DATA = '/mnt/g/Hemorrhage/00_Kaggle_RSNA/data_png/Normal/NKG________000640/'



# ===================================== import
import cv2, numpy as np, json
import pandas as pd
import os, sys
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import data_load_functions as dl
import models.model_definition as md

from utils.save_img import save_img
from utils.helper import *

# ============================================ Function    
def put_text(image,str_):
    import numpy as np
    from PIL import ImageFont, ImageDraw, Image

    image_ = Image.fromarray(np.uint8(image*255))
    font = ImageFont.truetype('../fonts/NanumBarunGothic.ttf',20)
    draw = ImageDraw.Draw(image_)
    draw.text((10,10),str_, font=font, fill=(255,255,255,0))
    image = np.float32(image_)/255
    return image




# ===================================== Init
set_gpu(GPU_ID)
    
    

# ====================================== load data
# 딱 한명분 로드 함
images, paths = dl.get_A_subject(DIR_INPUT_DATA) 
# images - (36, 512, 512, 3)


# ======================================= prepare tensorflow and keras
# 텐서플로 준비하는 코드를 삽입합니다. 
# 모델 준비, 모델 로드 포함
# 모델 정의는 models/model_definition.py 에.

### anomaly
model_ano = md.define_model_ano()
model_ano.load_weights(PATH_MODEL_ANO)
# model_ano.summary()
if SAVE_PB:
    tf.saved_model.save(model_ano,'../save_pb/ano')
    print('SAVE PB ano 완료')


### cls + seq
model_cls_seq = md.define_model_cls_seq()
model_cls_seq.load_weights(PATH_MODEL_CLS_SEQ)
model_cls_seq.summary()
if SAVE_PB:
    tf.saved_model.save(model_cls_seq,'../save_pb/cls_seq')
    print('SAVE PB cls_seq 완료')

    
# ======================================== pre processing
# 전처리 수행
# 히스토그램 평활화를 수행
# 추가적인 프로세스 있을수 있음
# 당장은 아무것도 하지 않음

print('Preprocessing ... ',end='')
# images_new = []
# for image in images:
# #     image = np.uint8(np.clip(np.squeeze(image*255),0,255))
#     image = np.uint8(np.clip(np.squeeze(image),0,255))
#     image = np.stack( [cv2.equalizeHist(image[:,:,c]) for c in range(3)], -1)
#     image = np.float32(image/255)
#     images_new.append( image)         
# images = np.stack(images_new) # [N, H, W, 3]
print('완료')
    
    

    
# ========================================= model 1
# 첫번째 모델 동작
# 여기서는 Anomaly detection

print('Anomaly Detection... ',end='')
preds_restored = []
for i in range(images.shape[0]):
    image = images[i:i+1,:,:,:]
    image = image[:,:,:,1:2] # 1채널만 작동하게
#     pred = model_ano(image*2-1)*0.5+0.5
    pred = model_ano(image)
    pred = np.array(pred)
    preds_restored.append(pred)

preds_restored = np.concatenate(preds_restored) # [N, H, W, 1]
preds_restored = np.clip( preds_restored, 0,1)
print('완료')



# ========================================= model 2+3
# 
# 여기서는 classification + sequntial network

print('Classification + Sequnce... ',end='')

images_ = [ cv2.resize(i,(256,256)) for i in images]
images_ = np.stack(images_)

preds_cls_slice, preds_seq_slice, preds_seq_patient, featuremaps = model_cls_seq( images_, training=True)

preds_cls_slice = np.reshape(preds_cls_slice,[-1])
preds_seq_slice = np.reshape(preds_seq_slice,[-1])
preds_seq_patient = np.reshape(preds_seq_patient,[-1])
featuremaps = np.float32(featuremaps)
print('완료...')





# ========================================= Postprocessing - CAM
# 후처리 CAM 
# 이 단계의 output
#   * 0~1 사이의 CAM 1개
#   * overlay = image + CAM(color) 1개
# 총 2개

print('Postprocessing-CAM...',end='')
# get last fc weight
denselayer = model_cls_seq.layers[1].layers[-2]
weights = denselayer.get_weights()[0]
#
cam_maps = []
cam_overlays = []

for image, featuremap, pred_slice in zip( images[:,:,:,1], featuremaps, np.squeeze(preds_seq_slice)):  
    ##################################################################
    def post_cam(image,feature, pred_slice):

        # cam 
        cam_map = np.maximum(np.dot(featuremap,weights),0)
#         cam_map = np.pad( cam_map, [[0,1],[0,1],[0,0]] )

        # 예쁘게 만들기
        k = cv2.getGaussianKernel(5,5/6);   k = k*k.T;   k = k/k.sum()
        cam_map = cv2.filter2D(cam_map,cv2.CV_32F,k)    
        cam_map = np.clip(cam_map/(cam_map.max()+1e-8)*(pred_slice*1.5),0,1)
        cam_map = np.clip(cv2.resize( cam_map, (512,512), interpolation=cv2.INTER_NEAREST ),0,0.5)

        # 컬러
        heatmap = cv2.applyColorMap(np.uint8(np.squeeze(cam_map)*255),cv2.COLORMAP_HOT)
        heatmap = np.float32(heatmap)*1/255

        # 오버레이
        i = np.tile(image.reshape([512,512,1]),[1,1,3])
        overlay = i*0.6+heatmap*0.4
        overlay = np.clip(np.flip(overlay,-1),0,1)
        
        return cam_map, overlay
    ###################################################################
    
    cam_map, overlay = post_cam(image,featuremap,pred_slice)
    
    #save
    cam_maps.append(cam_map)
    cam_overlays.append(overlay)
    
    if DEBUG:
    #     print(seq_slice)
        display('pred_slice', pred_slice)
        imshow(overlay*255)
        imshow(image*255)
print('완료')



# ========================================= Postprocessing - Anomaly
# 이 단계의 output
#   * 0~1 사이의 Anomaly map 1개
#   * overlay = image + AnoMap(color) 1개
# 총 2개
print('Postprcessing-Anomaly... ',end='')
ano_maps = []
ano_overlays = []
for image, restored, pred_slice in zip( images[:,:,:,1], np.squeeze(preds_restored), np.squeeze(preds_seq_slice)):
    
    #########################################################################
    def post_ano(image,restored):
        # anomaly region
        ano_region = make_ano_image_refine(image,restored)
        ano_region = np.clip(ano_region*1.5,0,1)
        
        # 블러커널 준비.
        k = cv2.getGaussianKernel(15,15/6)
        k = k*k.T
        k = k/np.sum(k)
        
        # 블러링
        ano_region = cv2.filter2D( ano_region, cv2.CV_32F, k )
        ano_region = ano_region/(ano_region.max()+1e-6) 
        ano_region = np.clip( ano_region,0,1)
        
        # pred_slice 가중치 
        ano_region = ano_region * pred_slice
        ano_region = np.clip( ano_region,0,1)        

        # colormap
        heatmap = cv2.applyColorMap(np.uint8(np.squeeze(ano_region)*255),cv2.COLORMAP_HOT)
        heatmap = np.float32(heatmap)*1/255

        # overlay
        i = np.tile(image.reshape([512,512,1]),[1,1,3])
        overlay = i*0.6+heatmap*0.4
        overlay = np.clip(np.flip(overlay,-1),0,1)
        
        return ano_region, overlay
    ##########################################################################
    
    ano_region, overlay = post_ano(image,restored)
    
    # save
    ano_maps.append(ano_region)
    ano_overlays.append(overlay)
    
    if DEBUG:
        print('pred_slice',pred_slice)
        imshow(image*255)
        imshow(ano_region*255)
        imshow(restored*255)
        imshow(overlay*255)
#     input()
print('완료')




# ========================================= Postprocessing - Score
# 이 단계의 output
#   * 0~1 사이로 변환환 Score 3가지 (scores_cnn_slice, scores_rnn_slice, score_patient
#   * 0 또는 1 표시되는 예측 결과 (hemopreds_cnn_slice, ~, ~)

print('Postprcessing-Score 조립(?)... ',end='')
scores_cnn_slice = np.reshape( ( preds_cls_slice ), [-1])
scores_seq_slice = np.reshape( ( preds_seq_slice ), [-1])
scores_seq_patient = np.reshape( ( preds_seq_patient ), [-1] )
hmpreds_cnn_slice = np.int32(scores_cnn_slice>0.5)
hmpreds_seq_slice = np.int32(scores_seq_slice>0.5)
hmpreds_seq_patient = np.int32(scores_seq_patient>0.5)

print('완료')

    
    

# ========================================== save - 중간결과
# 중간 영상 저장하는 코드
# seg 결과, cls CAM 결과, rnn slice 결과, rnn 환자 단위 결과
if SAVE_INTERMEDIATE :
    ### 저장할 환자 단위 폴더 생성
    DIR_OUTPUT_INTERMEDIATE = os.path.join( DIR_OUTPUT, 'intermediate')
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    os.makedirs(DIR_OUTPUT_INTERMEDIATE, exist_ok=True)

    ### 영상 저장
    print('영상 저장 ...',end='')
    filenames_input = [ os.path.splitext( os.path.split( p )[-1] )[0] for p in paths]
    for i in range(preds_restored.shape[0]):
        save_img(os.path.join(DIR_OUTPUT_INTERMEDIATE, f'original_{i:02d}.png'), images[i,:,:,1], verbose=0)
        save_img(os.path.join(DIR_OUTPUT_INTERMEDIATE, f'ano_restored_{i:02d}.png'), preds_restored[i], verbose=0)
        save_img(os.path.join(DIR_OUTPUT_INTERMEDIATE, f'ano_map_{i:02d}.png'), ano_maps[i], verbose=0)
        save_img(os.path.join(DIR_OUTPUT_INTERMEDIATE, f'ano_overlay_{i:02d}.png'), ano_overlays[i], verbose=0)
        save_img(os.path.join(DIR_OUTPUT_INTERMEDIATE, f'cam_{i:02d}.png'), cam_maps[i], verbose=0)
        save_img(os.path.join(DIR_OUTPUT_INTERMEDIATE, f'cam_overlay_{i:02d}.png'), cam_overlays[i], verbose=0)

        summary = np.hstack([
            np.stack([images[i,:,:,1]]*3,-1),
            np.stack([preds_restored[i,:,:,0]]*3,-1),
            put_text(np.stack([ano_maps[i]]*3,-1),f'socre_slice,patient : {scores_seq_slice[i]:.02f}|{scores_seq_patient[0]:.02f}' ),
            ano_overlays[i]
        ])
        summary = put_text(summary,filenames_input[i])
        imshow(summary*255)
    print('완료')
    

    ### rnn slice별 및 환자의 뇌출혈 유뮤 결과 저장
    # 길이 맞추기
    n_slices = scores_seq_slice.shape[0]
    scores_seq_patient_ = np.tile( scores_seq_patient, [n_slices] )
    hmpreds_seq_patient_ = np.tile( hmpreds_seq_patient, [n_slices] )

    tmp_ = np.stack([
        scores_seq_slice*100,
        scores_seq_patient_*100,
        hmpreds_seq_slice,
        hmpreds_seq_patient_,
    ],axis=-1)

    # np -> dataframe
    df = pd.DataFrame(
        tmp_,
        columns=[
            'scores_slice',
            'scores_patient',
            'hemorrhage_slice',
            'hemorrhage_patient'
        ]
    )
    
    ### save
    df.to_csv(os.path.join(DIR_OUTPUT_INTERMEDIATE, 'hm_scores.csv'))
    print('Slice hemorrhage score 저장완료...')


    
# ========================================== save - output
# output 저장하는 코드
# 영상 + JSON

### 디렉토리 생성
os.makedirs(DIR_OUTPUT, exist_ok=True)

### 결과 저장 - 영상
for out_image_, path, score_slice, hmpred_slice in zip(ano_overlays, paths, scores_seq_slice, hmpreds_seq_slice ):
    filename = os.path.split(path)[-1]
    save_img(os.path.join(DIR_OUTPUT, filename), out_image_, verbose=0)
    
### 결과 저장 - JSON
filenames_input = [ os.path.splitext( os.path.split( p )[-1] )[0] for p in paths]
save_json = {
    'hmpred_patient' : int(hmpreds_seq_patient),
    'score_patient' :  int(scores_seq_patient*100),
    'hmpred_slice' :   np.int32( hmpreds_seq_slice ).tolist(),
    'socre_patient' :  np.int32( scores_seq_slice*100 ).tolist(),
    'filenames_input' : filenames_input,
}

path_json = os.path.join(DIR_OUTPUT,'hm_result.json')
with open(path_json,'w') as f:
    json.dump( save_json, f, indent=4)





