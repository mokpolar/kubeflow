# /usr/bin/env python
import tensorflow as tf
import argparse
import os
import shutil

def save_model():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default = "/mnt/pv/models/keras/mobnet", type = str)
    args = parser.parse_args()

    model_path = args.model_path
    if not (os.path.isdir(model_path)):
        os.makedirs(model_path)

    model_file = os.path.join(model_path, 'model.pb')    
    
    print("Start training...")
    
    # Modeling area

    mobnet = tf.keras.applications.mobilenet
    model = mobnet.MobileNet(weights='imagenet')

    print("Training finished...")


    # Save model

    model.save(model_file, save_format='tf')

    print("Saved model to {}".format(model_path))

    shutil.copy(os.path.abspath(__file__), os.path.join(model_path, __file__))



if __name__ == '__main__':
    save_model()


