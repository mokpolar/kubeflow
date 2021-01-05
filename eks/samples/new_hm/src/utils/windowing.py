import numpy as np
from PIL import Image

def normalize(img, min_, max_):
    # min-max normalize
    # min_ = img.min()
    # max_ = img.max()
    #
    # if min_==max_:
    #     if max_==0:
    #         return img
    #     else:
    #         return np.uint8(img / max_ * 255.)

    img = (img-min_) / (max_-min_)
    img = np.uint8(img * 255.)

    return img

def get_re_window(img, level=40, width=100):

    if isinstance(img, Image.Image):
        img = np.array(img)

    # 기존 dicom의 level 40이 새로 뽑은 이미지 파일의 level 128
    REF_LEVEL = 128
    level = level - 40 + REF_LEVEL

    # get window min, max value
    window_min = level - width//2
    window_max = level + width//2
    assert window_min<window_max

    # re windowing
    img[img<window_min] = window_min
    img[img>window_max] = window_max

    # min max normalize
    img = normalize(img, window_min, window_max)

    return img

if __name__=='__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('./dataset/AJOU_test.csv')
    filepath = df.iloc[15]['path_ct']
    im = np.array(Image.open(filepath).convert('L'))

    plt.figure(figsize=(36, 12))
    plt.subplot(1,2,1)
    plt.imshow(im, 'gray')
    plt.subplot(1,2,2)
    im2 = get_re_window(im)
    plt.imshow(im2, 'gray')

    plt.show()