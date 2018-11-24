
import base64

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from models_inference.face_emotions_stream import FERStreamingModel

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

target_emotions = ['calm', 'anger', 'happiness']
model = FERStreamingModel(target_emotions, verbose=True)


@app.route('/analyse_emotions', methods=['POST'])
def analyse_emotions():
    """
    Analyse face emotions.
    """
    data = request.data.replace(b'data:image/jpeg;base64', b'')
    # FIXME hack to process base64 image (couldn't make it work directly)
    imgdata = base64.b64decode(data)
    img_array = np.frombuffer(imgdata, dtype=np.float64)
    result = model.predict(img_array)
    return jsonify(result)


@app.route('/locate_faces', methods=['POST'])
def locate_faces():
    """
    Locate faces.
    """
    data = request.data.replace(b'data:image/jpeg;base64', b'')
    # FIXME hack to process base64 image (couldn't make it work directly)
    imgdata = base64.b64decode(data)
    img_array = np.frombuffer(imgdata, dtype=np.float64)
    face_locations = locate_faces(img_array)
    results = {}
    for pos, face in enumerate(face_locations):
        results[pos] = {'bbox': [int(i) for i in face]}
    return jsonify(results)


if __name__ == '__main__':
    app.run()
