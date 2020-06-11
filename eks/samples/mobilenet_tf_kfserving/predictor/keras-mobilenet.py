import tensorflow as tf
import argparse
import os



def train():
    print("TensorFlow version: ", tf.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/mnt/pv/models/tensorflow/mobilenet', type=str)
    args = parser.parse_args()
    version = 1
    export_path = os.path.join(args.model_path, str(version))

    print("Training....")

    mobilenet = tf.keras.applications.mobilenet
    model = mobilenet.MobileNet(weights = "imagenet")

    print("Training finished...")

    # after tf 2.1, export_saved_model method is removed
    #tf.keras.experimental.export_saved_model(model, model_path)
    #tf.keras.models.save_model(model, model_path, save_format="tf")
    model.save(export_path)

    # save model
    print("Saved model to {}".format(export_path))

if __name__=="__main__":
    train()
