from PIL import Image
import numpy as np
import pandas as pd
import os
import natsort
from utils.windowing import get_re_window



import glob, cv2
def get_A_subject(dir_):
    paths = glob.glob(dir_+'/**/*.png',recursive=True)
#     paths = sorted(paths)
    paths = natsort.natsorted(paths)
    images = [cv2.imread(p,cv2.IMREAD_GRAYSCALE) for p in paths]
    images = [cv2.resize(i,(512,512)) for i in images]
    images = [get_re_window(i) for i in images]
    # 3slice로 맞춤. 맨앞과 맨 뒤는 중복해서 출현
    A = [images[0]] + images[:-1]
    B = images
    C = images[1:] + [images[-1]]
    images = np.stack([np.stack([a,b,c],-1) for a,b,c in zip(A,B,C)])
    
    
    images = np.float32(images)/255
    return images, paths

