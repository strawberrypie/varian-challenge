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
from skimage import measure

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
    # xs = preprocess.process_test(directory)


    l = np.load('varian/services/380677.npz')
    xs = l['X']

    ys, percents = do_all_the_magic(xs)

    pictures = make_pictures(filepath, xs, ys)
    result = [{"percent": p, "image": i} for (p, i) in zip(list(percents), pictures)]
    return jsonify(result)


def do_all_the_magic(xs):
    l = np.load('varian/services/380677.npz')
    # ys, percents = some_magic(xs)
    ys = l['Y']
    percents = np.random.rand(len(xs))

    return ys, percents


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


def make_pictures(filepath, xs, ys):
    result_folder = os.path.join(RESULT_FOLDER, '{}-dir'.format(os.path.basename(filepath)))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    results = []
    for i, (xs_item, ys_item) in enumerate(zip(xs, ys)):
        filename = os.path.join(result_folder, '{}.png'.format(i))

        saved = beauty_image(filename, xs_item, ys_item)

        # plt.imsave(filename, xs_item, cmap=plt.get_cmap('gray'))


        with open(saved, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            results.append(encoded)

    return results

def beauty_image(filename, xs, ys):
    contours = measure.find_contours(ys, 0.8)
    xs_new = np.array(xs / xs.max(), dtype=np.float32)
    for contour in contours:
        for (x, y) in contour:
            iy = int(y)
            ix = int(x)
            xs_new.itemset((ix, iy), 1.0)
            xs_new.itemset((ix + 1, iy), 1.0)
            xs_new.itemset((ix - 1, iy), 1.0)
            xs_new.itemset((ix, iy - 1), 1.0)
            xs_new.itemset((ix, iy + 1), 1.0)

    plt.imsave(filename, xs_new, cmap=plt.get_cmap('gray'))
    return filename
    #
    # fig, ax = plt.subplots(figsize=(15, 15))
    # # ax.imshow(xs_item * ALPHA + (1 - ALPHA) * ys_item * xs_item, cmap=plt.get_cmap('gray'))
    # ax.imshow(xs, cmap=plt.get_cmap('gray'))
    #
    # contours = measure.find_contours(ys, 0.8)
    # for n, contour in enumerate(contours):
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=4)
    #
    # fig.savefig(filename)
    # return filename