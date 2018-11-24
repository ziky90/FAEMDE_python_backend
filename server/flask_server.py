
import base64

import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

from models_inference.face_emotions_stream import FERStreamingModel
from models_inference.locate_faces import locate_faces

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

target_emotions = ['calm', 'anger', 'happiness']
model = FERStreamingModel(target_emotions, verbose=True)


def read_image():
    """
    Hacky way to read the image.
    :return: Loaded image.
    :rtype: np.ndarray
    """
    data = request.data.replace(b'data:image/jpeg;base64', b'')
    imgdata = base64.b64decode(data)
    filename = 'image_to_process.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    image = cv2.imread(filename)
    return image


@app.route('/analyse_emotions', methods=['POST'])
def analyse_emotions():
    """
    Analyse face emotions.
    """
    image = read_image()
    result = model.predict(image)
    return jsonify(result)


@app.route('/find_faces', methods=['POST'])
def find_faces():
    """
    Locate faces.
    """
    image = read_image()
    face_locations = locate_faces(image)
    results = {}
    for pos, face in enumerate(face_locations):
        results[pos] = {'bbox': [int(i) for i in face]}
    return jsonify(results)


if __name__ == '__main__':
    app.run()
