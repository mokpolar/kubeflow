import numpy as np
import cv2

def get_binary_CAM(img, feature_map, weights):
    """
    Parmas:
        img         [512, 512]
        weights     [1024,1]
        feature_map [16, 16, 1024]
    Returns:
        heatmap     [512, 512, 3]
        overlay_img [512, 512, 3]
    """
    

    # Multiple cam feature map and binary weights(0~1)
    cam = np.dot(feature_map, weights)

    # normalize
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = np.uint8(cam * 255)
    
    # resize
    cam = cv2.resize(cam, (512, 512))

    # get heatmap
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    
    # get overlay image
    overlay_img = np.uint8(img * 255 * 0.6 + heatmap * 0.4)
    
    return heatmap, overlay_img
    