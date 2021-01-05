import matplotlib.pyplot as plt
import numpy as np

def save_img(save_dir, img, cmap='gray', verbose=0):
    
    if len(img.shape)>2:
        
        # gray -> dimension reduction
        if img.shape[-1]==1:
            img = np.squeeze(img)
                    
    plt.imsave(save_dir, img, cmap=cmap)
    if verbose==1:
        print(f'Image was saved in [{save_dir}]')

