import base64
import pickle
import threading
from flask import Flask, request, Response, jsonify

from visual_slam import start_slam, get_frames

app = Flask(__name__)


@app.route('/slam/start', methods=['POST'])
def start():
    # Start SLAM in a new thread
    thread = threading.Thread(target=start_slam)
    thread.start()

    return Response(status=204)


def serialize_frames(frames):
    serialized_pred = pickle.dumps(frames)
    base64_pred = base64.b64encode(serialized_pred).decode('utf-8')

    return jsonify(serialized_data=base64_pred)


@app.route('/slam/get_last_frame', methods=['POST'])
def get_last_frame():
    frames = get_frames(idx=-1)

    return serialize_frames(frames)


@app.route('/slam/get_frame_by_idx', methods=['POST'])
def get_frame_by_idx():
    data = request.json
    idx = data['idx']
    frames = get_frames(idx=idx)

    return serialize_frames(frames)


@app.route('/slam/get_all_frames', methods=['POST'])
def get_all_frames():
    frames = get_frames()

    return serialize_frames(frames)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=6000, use_reloader=False)
