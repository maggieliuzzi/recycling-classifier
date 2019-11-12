import numpy as np
import os
import ssl
import argparse
from PIL import ImageFile  # For solving issue: "IOError: image file is truncated (19 bytes not processed)"
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
# Ensures that Keras is using TensorFlow as a back-end
keras_path = os.path.join(os.path.expanduser('~'), '.keras')
keras_json_path = os.path.join(keras_path, 'keras.json')
if not os.path.isdir(keras_path):
    os.makedirs(keras_path)
with open(keras_json_path, 'w') as kf:
    contents = "{\"epsilon\": 1e-07,\"image_data_format\": \"channels_last\",\"backend\": \"tensorflow\",\"floatx\": \"float32\"}"
    kf.write(contents)
'''

from keras.applications.mobilenetv2 import MobileNetV2
from keras import models, layers
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser(
    description="Retrains top layers of MobileNetV2 for classifying images into various materials for accurate recycling.",
    epilog="")
parser.add_argument('-e', '--epochs', default=3, help="the number of epochs to train the network for; required.")
args = parser.parse_args()

'''
+-----------------------+
| MobileNetV2 structure |
+-----------------------+

Input                   (224, 224, 3)
ZeroPadding2D           (225, 225, 3)
<BLOCK 0>
<BLOCK 1>                
<BLOCK 2>
...
<BLOCK 15>
<BLOCK 16>
Conv2D                  (7, 7, 1280)
BatchNormalization+ReLu (7, 7, 1280 each)
GlobalAveragePooling2D  (1280)
Dense                   (1000)

Where <BLOCK> consists of:
Conv2D
BatchNormalization+ReLU
DepthwiseConv2D
BatchNormalization+ReLU
Conv2D
BatchNormalization

Structure described here: http://machinethink.net/blog/mobilenet-v2/

The final Conv2D layer and onwards are the target of re-training
The final Dense layer is replaced with a new one of size (2) for classifying good and faulty.
'''

np.random.seed(3)

home_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(home_path, "training-data-formatted/")
train_path = os.path.join(dataset_path, "train")
validate_path = os.path.join(dataset_path, "validate")

# Download MobileNetV2 without its classifier and freeze all but the last 4 layers
ssl._create_default_https_context = ssl._create_unverified_context
model_mobile = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), include_top=False, pooling='avg')
for layer in model_mobile.layers[:-4]:
    layer.trainable = False

# Add new classifier for 6 classes "Cardboard", "Glass", "Metal", "Paper", "Plastic" and "Trash"
model_new = models.Sequential()
model_new.add(model_mobile)
model_new.add(layers.Dense(6, activation='softmax'))
model_new.summary()

# Define the data generators, including random data transforms as it is being input
train_trans = ImageDataGenerator(rescale=1. / 255, rotation_range=36, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
validate_trans = ImageDataGenerator(rescale=1. / 255)
train_generator = train_trans.flow_from_directory(train_path, target_size=(224, 224), batch_size=80, seed=3, class_mode='categorical')
validate_generator = validate_trans.flow_from_directory(validate_path, target_size=(224, 224), batch_size=16, seed=3, class_mode='categorical', shuffle=False)

# Train the neural network
model_new.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
print(train_generator)
print(validate_generator)

models_path = os.path.join(home_path,'trained_models')
if not os.path.isdir(models_path):
    os.makedirs(models_path)

model_new.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(models_path+"/model.{epoch:02d}.hdf5", verbose=1)
history = model_new.fit_generator(train_generator, epochs=int(args.epochs), steps_per_epoch=len(train_generator), verbose=1,
                        validation_data=validate_generator, validation_steps=len(validate_generator), callbacks=[checkpointer])
print("Finished training and saving models.")