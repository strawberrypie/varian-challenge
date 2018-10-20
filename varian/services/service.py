import base64
import os
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename, redirect
import numpy as np
import zipfile
from src import preprocess
from matplotlib import pyplot as plt

app = Flask(__name__)
CORS(app)

UPLOADER_FOLDER = '/tmp/upload'
UNZIP_FOLDER = '/tmp/unzip'
RESULT_FOLDER = '/tmp/results'

EXAMPLE='varian/services/380677.npz'

@app.route('/v1/ping')
def ping():
    l = np.load(EXAMPLE)
    x = l['X']
    y = l['Y']

    print(x[0].shape)

    # print(len(l))
    return 'Pong!'

@app.route('/v1/predict', methods=['POST'])
def predict():
    ### read
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)

    file = request.files['file']

    filepath = save(file)
    directory = unzip(filepath)

    # here preprocess and so on
    #xs = preprocess.process_test(directory)

    l = np.load('varian/services/380677.npz')
    xs = l['X']
    xs_new = np.array(xs / xs.max(), dtype=np.float32)


    results = make_pictures(filepath, xs_new)
    print(results)
    return jsonify(results)


def save(file):
    filename = secure_filename(file.filename)
    if not os.path.exists(UPLOADER_FOLDER):
        os.makedirs(UPLOADER_FOLDER)

    filepath = os.path.join(UPLOADER_FOLDER, filename)
    file.save(filepath)

    return filepath

def unzip(filepath):
    zip_ref = zipfile.ZipFile(filepath, 'r')

    directory = os.path.join(UNZIP_FOLDER, '{}-dir'.format(os.path.basename(filepath)))

    zip_ref.extractall(directory)
    zip_ref.close()

    files = [f for f in os.listdir(directory) if not f.startswith(".")]

    return os.path.join(directory, files[0])

# later xs_old, xs, ys
def make_pictures(filepath, xs_old):
    print(xs_old.shape)
    result_folder = os.path.join(RESULT_FOLDER, '{}-dir'.format(os.path.basename(filepath)))
    print(result_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    results = []
    for i, xs_item in enumerate(xs_old):
        filename = os.path.join(result_folder, '{}.png'.format(i))

        plt.imsave(filename, xs_item, cmap=plt.get_cmap('gray'))

        with open(filename, 'rb') as f:
            encoded = str(base64.b64encode(f.read()))
            results.append(encoded)

    return results