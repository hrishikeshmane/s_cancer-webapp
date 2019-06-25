import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from pathlib import Path

from skimage.io import imread
from skimage.transform import resize

import cv2
import numpy as np

from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.utils import to_categorical

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

def top_2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true=y_true, y_pred=y_pred, k=2)

model=tf.keras.models.load_model('./models/sc_best_model.h5', custom_objects={'top_2': top_2})
print("model loaded.... Check http://127.0.0.1:5000/")

#img= Path("sc_test/benign/729.jpg")
#650
#100,101,102,103,144
img_height, img_width, img_channels = 224,224,3
batch_size = 64
nb_classes = 2


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(img_height, img_width))

    test_images=[]
    test_labels=[]
    img = imread(img)
    img = cv2.resize(img, (img_height, img_width))
    test_images.append(img)

    test_images = np.array(test_images, dtype=np.float32)
    test_images = mobilenet_v2.preprocess_input(test_images)
    test_labels = np.array(test_labels)
    test_labels_cat = to_categorical(test_labels, num_classes=2)

    preds = model.predict(test_images)
    preds = np.argmax(preds, axis=-1)

    return preds


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

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        if (preds[0]==1):
        	result="malignant"
        else:
        	result="benign"
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
