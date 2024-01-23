from tensorflow import keras
from keras.preprocessing import image
# from PIL import Image
import numpy as np

def preprocess_image(image_path) :
    img = image.load_img(image_path, target_size=(240,240))
    
    img = image.img_to_array(img)
    
    img = np.expand_dims(img, axis=0)
    
    return img

def get_prediction(image_path) :
    img = preprocess_image(image_path)
    
    model = keras.models.load_model("model_classification")
    
    predictions = model.predict(img)
    
    return predictions