from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import string

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Model/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

symbols = string.ascii_lowercase + "0123456789" # All symbols captcha can contain
num_symbols = len(symbols)
img_shape = (50, 200, 1)

def model_predict(img_path, model):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0
    else:
        print("Not detected");
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (5, 36))
    l_ind = []
    probs = []
    for a in ans:
        l_ind.append(np.argmax(a))
        #probs.append(np.max(a))

    capt = ''
    for l in l_ind:
        capt += symbols[l]
    return capt


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = img / 255.0
        else:
            print("Not detected");
        # Make prediction
        res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
        ans = np.reshape(res, (5, 36))
        l_ind = []
        probs = []
        for a in ans:
        
            l_ind.append(np.argmax(a))
        #probs.append(np.max(a))

        capt = ''
        for l in l_ind:
            capt += symbols[l]

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(capt, top=1)   # ImageNet Decode
            #result = str(capt)               # Convert to string
        return capt
    return None


if __name__ == '__main__':
    app.run(debug=True)