import time
from camera import PinholeCamera
from dataset import VideoDataset, Dataset
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes
from feature_tracker_configs import FeatureTrackerConfigs
from slam import Slam
import threading
from viewer3D import Viewer3D


class VisualSLAMSystem:
    def __init__(self, num_features=2000, tracker_type=FeatureTrackerTypes.DES_BF):
        tracker_config = FeatureTrackerConfigs.TEST
        tracker_config['num_features'] = num_features
        tracker_config['tracker_type'] = tracker_type
        self.feature_tracker = feature_tracker_factory(**tracker_config)
        self.slam = None
        self.dataset = None
        self.cam = None
        self.is_running = False

    def start(self, cam: PinholeCamera, dataset: Dataset):
        self.cam = cam
        self.dataset = dataset

        self.slam = Slam(self.cam, self.feature_tracker, None)

        thread = threading.Thread(target=self._run_slam)
        thread.start()

        self.is_running = True

    def _run_slam(self):
        viewer3D = Viewer3D()

        img_id = 0
        while self.dataset.isOk():
            print(f'Processing image: {img_id}')

            img = self.dataset.getImageColor(img_id)

            if img is None:
                break

            timestamp = self.dataset.getTimestamp()
            next_timestamp = self.dataset.getNextTimestamp()
            frame_duration = next_timestamp - timestamp

            time_start = time.time()
            self.slam.track(img, img_id, timestamp)
            duration = time.time() - time_start

            viewer3D.draw_map(self.slam)

            if frame_duration > duration:
                time.sleep(frame_duration - duration)

            img_id += 1

        self.stop()

    def stop(self):
        if self.slam:
            self.slam.quit()
            self.is_running = False

    def get_frames(self, idx=None):
        frames = []

        if idx is None:
            for idx in range(len(self.slam.map.frames)):
                frame = self._extract_frame_data(idx)
                frames.append(frame)
        else:
            frame = self._extract_frame_data(idx)
            return [frame]

        return frames

    def _extract_frame_data(self, idx):
        frame = {}
        frame_data = self.slam.map.frames[idx]
        good_indexes = [i for i, p in enumerate(frame_data.points) if p is not None and not p.is_bad]

        frame['points'] = [frame_data.points[i].pt for i in good_indexes]
        frame['kps'] = [frame_data.kps[i] for i in good_indexes]
        frame['pose'] = frame_data.Twc
        frame['id'] = frame_data.id

        return frame


if __name__ == "__main__":
    slam_system = VisualSLAMSystem()

    cam_intrinsics = {
        'width': 3840,
        'height': 2160,
        'fx': 2376.5625,
        'fy': 2376.5625,
        'cx': 1920,
        'cy': 1080,
        'DistCoef': [0.13, -0.24, 0.0, 0.0, 0.0],
        'fps': 59.96
    }

    from config import Config
    from dataset import dataset_factory

    config = Config()
    dataset = dataset_factory(config.dataset_settings)

    cam = PinholeCamera(config.cam_settings['Camera.width'],
                        config.cam_settings['Camera.height'],
                        config.cam_settings['Camera.fx'],
                        config.cam_settings['Camera.fy'],
                        config.cam_settings['Camera.cx'],
                        config.cam_settings['Camera.cy'],
                        config.DistCoef,
                        config.cam_settings['Camera.fps']
                        )

    slam_system.start(cam=cam, dataset=dataset)
