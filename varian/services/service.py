import base64
import os
import tempfile

import dicom
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename, redirect
import numpy as np
import zipfile
from src import preprocess
from matplotlib import pyplot as plt
from skimage import measure
from varian.services.all_magic import perform_cv
from varian.services.another_magic import mk_mask
import logging

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

    filepath = save_1(file)
    directory = unzip(filepath)
    files = [f for f in os.listdir(directory) if not f.startswith(".") and "MR." in f]
    xs = np.array([dicom.read_file(os.path.join(directory, f)).pixel_array for f in files], dtype=np.uint16)
    # here preprocess and so on
    # xs = preprocess.process_test(directory)


    logging.info("loading...")
    # l = np.load('varian/services/380677.npz')
    # l = np.load('varian/services/ex_1.npz')
    # xs = l['X']
    logging.info("processing...")
    ys, masks, percents = do_all_the_magic(xs)
    logging.info("making pictures...")
    # pictures = make_pictures(filepath, xs, ys)

    fusks = [to_base64(save(x)) for x in xs]
    result = [{"percent": p, "image": i, "processed": pr, "mask": m} for (p, i, pr, m) in zip(list(percents), fusks, ys, masks)]
    return jsonify(result)


def do_all_the_magic(xs):
    l = np.load('varian/services/380677.npz')
    # l = np.load('varian/services/ex_1.npz')
    ys = []
    masks = []
    for i, xs_item in enumerate(xs):
        logging.info("perform {}".format(i))

        ys_arr = perform_cv(xs_item.astype(np.uint16))


        print("!!!!!!!!!", type(ys_arr))

        ys.append(to_base64(save(ys_arr)))
        logging.info("making mask {}".format(i))
        # mask_arr = mk_mask(xs_item)
        mask_arr = perform_cv(xs_item.astype(np.uint16))

        masks.append(to_base64(save(mask_arr)))

    # ys = np.array([perform_cv(xs_item.astype(np.uint16)) for xs_item in xs])
    # ys = np.array(ys)
    # masks = np.array(masks)
    # ys = l['Y_pred']
    # ys = l['Y']
    percents = list(np.random.rand(len(xs)))

    return ys, masks, percents


def save_1(file):
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

def save(n):
    result_folder = os.path.join(RESULT_FOLDER)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    logging.info(type(n))
    filename = os.path.join(result_folder, 'temp.png')
    plt.imsave(filename, n, cmap=plt.get_cmap('gray'))
    return filename


def to_base64(filename):
    with open(filename, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return encoded


def beauty_image(filename, xs, ys):
    logging.info("making pictire for rile {}...".format(filename))
    contours = measure.find_contours(ys, 0.8)
    xs_new = np.array(xs / xs.max(), dtype=np.float32)

    overlay = np.zeros(xs_new.shape)

    #new version
    # for contour in contours:
    #     cx = [x for (x, y) in contour]
    #     cy = [y for (x, y) in contour]
    #     center = np.array([np.mean(cx), np.mean(cy)])
    #     cr = np.max([np.max(cx) - np.min(cx), np.max(cy) - np.min(cy)])
    #
    #     for p, _ in np.ndenumerate(overlay):
    #         density = max(0, (cr - np.linalg.norm(p - center)) / cr)
    #         overlay.itemset(p, density)
    #
    # ALPHA = 0.2
    # plt.imsave(filename, xs_new * ALPHA + (1 - ALPHA) * xs_new * overlay, cmap=plt.get_cmap('gray'))
    # return filename
    #old version

    # old version
    for contour in contours:
        for (x, y) in contour:
            iy = int(y)
            ix = int(x)
            xs_new.itemset((ix, iy), 1.0)
            # xs_new.itemset((ix + 1, iy), 1.0)
            # xs_new.itemset((ix - 1, iy), 1.0)
            # xs_new.itemset((ix, iy - 1), 1.0)
            # xs_new.itemset((ix, iy + 1), 1.0)

    plt.imsave(filename, xs_new, cmap=plt.get_cmap('gray'))
    return filename
    # end old version
