import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename, redirect
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOADER_FOLDER='/tmp/upload'

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
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)

    file = request.files['file']
    filename = secure_filename(file.filename)

    if not os.path.exists(UPLOADER_FOLDER):
        os.mknod(UPLOADER_FOLDER)

    filepath = os.path.join(UPLOADER_FOLDER, filename)

    file.save(filepath)
    # print(r)
    # content = request.json
    print(5)

    # res = predict(content)
    return jsonify(len([5]))


def predict(req):
    response = req
    response['data'] = [{'info': 'abracadabra', 'source': req['data'][0]}]
    return response