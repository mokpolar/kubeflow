import GPUtil,os

def set_gpu(gpu_id='AUTO'):
    '''  gpu_id : 'AUTO' : 자동  '1' : 1번 GPU  'CPU' : CPU   '''
    if gpu_id == 'AUTO':       gpu_id = str(GPUtil.getAvailable()[0])
    if gpu_id == 'CPU' :       gpu_id = '-1'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return gpu_id



import cv2
def make_brain_mask(images_1,images_2):
        
    black1 = images_1<(10/255)
    black2 = images_2<(10/255)
    white1 = images_1>(250/255)
    white2 = images_2>(250/255)
    
    mask = 1-(black1|black2|white1|white2)
#     kernel = np.ones((7, 7), np.uint8)
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.erode(np.uint8(np.squeeze(mask)), kernel7, iterations=2)
    mask = cv2.dilate(mask, kernel7, iterations=1)
    mask = cv2.dilate(mask, kernel3, iterations=1)
    return mask

def make_ano_image(image, restored_image):
    ''' 
    image : CT
    restored_iamge : Ano 네트워크에서 복원한 이미지
    '''
    diff = np.maximum(image-restored_image-0/255,0)
    return diff

def make_ano_image_refine(image, restored_image):
    ''' 
    image : CT
    restored_iamge : Ano 네트워크에서 복원한 이미지
    '''    
    mask = make_brain_mask(image,restored_image)
    diff = np.maximum(image-restored_image-0/255,0)*mask
#     diff = np.maximum(np.abs(image-restored_image)-0/255,0)*mask  ## 실험중
    diff = np.uint8(diff*255)
    kernel = np.ones((3, 3), np.uint8)
    diff = cv2.erode(np.uint8(np.squeeze(diff)), kernel, iterations=2)
    diff = cv2.dilate(diff, kernel, iterations=2)
    diff = np.clip(np.float32(diff)/255,0,1)
    return diff


import numpy as np
from PIL import Image
def imshow(image):
    from PIL import Image
    display(Image.fromarray(np.squeeze(np.uint8(image))))


    
import numpy as np
# def sigmoid(M):
#     M = 1/(1+np.exp(-M))
#     return M

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

