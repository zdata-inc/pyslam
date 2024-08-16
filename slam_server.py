import base64
import pickle
from flask import Flask, request, Response, jsonify
from visual_slam import VisualSLAMSystem
from dataset import VideoDataset, DatasetType
from camera import PinholeCamera

app = Flask(__name__)
slam_system = VisualSLAMSystem()

curr_frame_idx = 0


def serialize_frames(frames):
    serialized_pred = pickle.dumps(frames)
    base64_pred = base64.b64encode(serialized_pred).decode('utf-8')

    return jsonify(serialized_data=base64_pred)


@app.route('/slam/start', methods=['POST'])
def start_slam():
    data = request.json
    cam_intrinsics = data['camera_intrinsics']
    video_dir = data['video_dir']
    video_filename = data['video_filename']

    cam = PinholeCamera(cam_intrinsics['width'],
                        cam_intrinsics['height'],
                        cam_intrinsics['fx'],
                        cam_intrinsics['fy'],
                        cam_intrinsics['cx'],
                        cam_intrinsics['cy'],
                        cam_intrinsics['DistCoef'],
                        cam_intrinsics['fps']
                        )

    dataset = VideoDataset(path=video_dir, name=video_filename, type=DatasetType.VIDEO)

    slam_system.start(cam=cam, dataset=dataset)

    return Response(status=204)


@app.route('/slam/is_running', methods=['POST'])
def is_running():
    running = slam_system.is_running

    return jsonify(is_running=running)


@app.route('/slam/get_last_frame', methods=['POST'])
def get_last_frame():
    frames = slam_system.get_frames(idx=-1)

    return serialize_frames(frames)


@app.route('/slam/get_new_frames', methods=['POST'])
def get_new_frames():
    global curr_frame_idx

    frames = slam_system.get_new_frames(start_idx=curr_frame_idx)

    curr_frame_idx += len(frames)

    return serialize_frames(frames)


@app.route('/slam/get_all_frames', methods=['POST'])
def get_all_frames():
    frames = slam_system.get_frames()

    return serialize_frames(frames)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=6000, use_reloader=False)
