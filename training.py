import numpy as np
import os
import ssl
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(
    description="Trains a convolutional neural network for detecting object material using ImageNet.")
parser.add_argument('--epochs', default=1, help="the number of epochs to train the network for.")
parser.add_argument('--theano', action='store_true', help="Use Theano as backend")
args = parser.parse_args()

if args.theano:
    # Ensures Keras is using Theano as a back-end, then loads Keras
    keras_path = os.path.join(os.path.expanduser('~'), '.keras')
    keras_json_path = os.path.join(keras_path, 'keras.json')
    if not os.path.isdir(keras_path):
        os.makedirs(keras_path)
    with open(keras_json_path, 'w') as kf:
        contents = "{\"epsilon\": 1e-07,\"image_data_format\": \"channels_last\",\"backend\": \"theano\",\"floatx\": \"float32\"}"
        kf.write(contents)

from keras.applications.mobilenetv2 import MobileNetV2
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator

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
'''

np.random.seed(1337)

home_path = os.path.dirname(__file__)
train_path = os.path.join(home_path, "recycling_dataset/train")
validate_path = os.path.join(home_path, "recycling_dataset/validate")
test_path = os.path.join(home_path, "recycling_dataset/test")

# Download MobileNetV2 without its classifier and freeze all but the last 4 layers
ssl._create_default_https_context = ssl._create_unverified_context
model_mobile = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), include_top=False, pooling='avg')
for layer in model_mobile.layers[:-4]:
    layer.trainable = False

# Build the new model
model_new = models.Sequential()
model_new.add(model_mobile)
bucket_size = 10 # TODELETE
model_new.add(layers.Dense(int(60/bucket_size), activation='softmax'))
model_new.summary()

# Define the data generators, including random data transforms as it is being input
train_trans = ImageDataGenerator(rescale=1. / 255, rotation_range=36, width_shift_range=0.2,
                                 height_shift_range=0.2, horizontal_flip=True)
validate_trans = ImageDataGenerator(rescale=1. / 255)
train_generator = train_trans.flow_from_directory(train_path, target_size=(224, 224), batch_size=80, seed=3,
                                                  class_mode='categorical')
validate_generator = validate_trans.flow_from_directory(validate_path, target_size=(224, 224),
                                                        batch_size=16, seed=3, class_mode='categorical',
                                                        shuffle=False)

# Train the neural network
model_new.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
print(train_generator)
print(validate_generator)

model_new.fit_generator(train_generator, epochs=int(args.epochs), steps_per_epoch=len(train_generator), verbose=1,
                        validation_data=validate_generator, validation_steps=len(validate_generator))
model_new.save('final_model_'+datetime.today().strftime('%d-%m-%Y_%H:%M:%S')+'.h5')
