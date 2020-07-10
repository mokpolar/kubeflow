from PIL import Image
import numpy as np
import pandas as pd
import os

def get_A_subject(df_path, img_dir, patient_id):
    '''
    return (np.array)(N,H,W,C) : N=약 30장, H=W=512, C=1
           (float32)
    '''
    # read df
    df = pd.read_csv(df_path)
    
    # get patient ids
    uids = df.study_instance_uid.unique()
    
    # patient ID 의 환자 불러오기
    tmp_df = df[df.study_instance_uid == uids[patient_id]]
    
    imgs = []
    for fn in tmp_df.filename:
        
        img_path = os.path.join(img_dir, fn+'.png')
        
        img = Image.open(img_path)
        imgs.append(np.array(img))
        
    return np.array(imgs)
