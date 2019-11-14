# This script defines functions needed to make predictions with the trained material-classifying model

from PIL import Image
import numpy as np
import io

from keras.applications.mobilenetv2 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model

def prepare_model(model_path):
    # Loads the keras model specified by model_path and returns the loaded model
    model = load_model(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model._make_predict_function()
    return model

def predict_from_file(loaded_model, pic_path):
    # Makes prediction against loaded_model for an image specified by pic_path and returns the probability vector
    # Use prepare_model first to create the loaded model
    img = image.load_img(pic_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predict = loaded_model.predict(img)
    return predict[0]

def predict_from_pil(loaded_model, pil_file):
    # Makes prediction against loaded_model for an image provided as a PIL object and returns the probability vector
    # Use prepare_model first to create the loaded model
    img = Image.open(io.BytesIO(pil_file))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predict = loaded_model.predict(img)
    return predict[0]
