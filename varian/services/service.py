from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/v1/ping')
def ping():
    return 'Pong!'

@app.route('/v1/predict', methods=['POST'])
def predict():
    content = request.json

    res = predict(content)
    return jsonify(res)


def predict(req):
    response = req
    response['data'] = [{'info': 'abracadabra', 'source': req['data'][0]}]
    return response