# KFServing with MobileNet on aws

## Workflow
* Set Environmetn using [Makefile](https://github.com/mokpolar/kubeflow/blob/master/eks/Makefile)
    * create eks cluster & nodegroup
    * install kubeflow on nodegroup

* build pv & binding pvc

* fairing
    * make dockerfile
    * model code
    * docker

* KFserving
    * transformer-preprocessing
    * predictor
    * transformer-postprocessing
    * explainer

---

## Workflow

```py
import tensorflow as tf
print('tensorflow: ', tf.__version__)

mobnet = tf.keras.applications.mobilenet
pre_processing_fn = mobnet.preprocess_input
post_processing_fn = mobnet.decode_predictions

model = mobnet.MobileNet(weights='imagenet')
model.save('./mobilenet_saved_model', save_format='tf')

image_saved_path = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
)


# Preprocessing
def _load_b64_string_to_img(b64_byte_string):
        image_bytes = base64.b64decode(image_byte_string)
        image_data = BytesIO(image_bytes)
        img = Image.open(image_data)
        return img

def preprocess_fn(instance):
    img = _load_b64_string_to_img(instance['image_bytes'])
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # The inputs pixel values are scaled between -1 and 1, sample-wise.
    x = tf.keras.applications.mobilenet.preprocess_input(
        img_array,
        data_format='channels_last',
    )
    x = np.expand_dims(x, axis=0)
    return x


# Load a model
model = tf.saved_model.load('./mobilenet_saved_model')


# Postprocessing ----------------------------
def postprocess_fn(pred):
    decoded = tf.keras.applications.mobilenet.decode_predictions(
        [pred], top=5
    )
    return decoded
```


* Get Saved_Model for predictor

get_saved_model.py

```py
import tensorflow as tf
print('tensorflow: ', tf.__version__)

mobnet = tf.keras.applications.mobilenet

model = mobnet.MobileNet(weights='imagenet')
model.save('./mobilenet_saved_model', save_format='tf')
preprocessing = mobnet.preprocess_input
post_processing = mobnet.decode_predictions

```


* Create a transformer for pre-processing and post-processing

```docker
docker build -t pydemia/mobilenet_transformer:latest -f transformer.Dockerfile .
docker tag pydemia/mobilenet_transformer:latest gcr.io/ds-ai-platform/mobilenet_transformer:latest
#docker push gcr.io/ds-ai-platform/mobilenet_transformer:latest
```