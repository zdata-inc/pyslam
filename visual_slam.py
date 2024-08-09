#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import time

from camera import PinholeCamera
from config import Config
from dataset import dataset_factory
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes
from feature_tracker_configs import FeatureTrackerConfigs
from slam import Slam

slam = None

def start_slam():
    config = Config()

    dataset = dataset_factory(config.dataset_settings)

    cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                        config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                        config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                        config.DistCoef, config.cam_settings['Camera.fps'])

    num_features = 2000

    tracker_type = FeatureTrackerTypes.DES_BF
    tracker_config = FeatureTrackerConfigs.TEST
    tracker_config['num_features'] = num_features
    tracker_config['tracker_type'] = tracker_type
    feature_tracker = feature_tracker_factory(**tracker_config)

    # create SLAM object
    slam = Slam(cam, feature_tracker, None)

    # create Viewer object
    # viewer3D = Viewer3D()

    img_id = 0  #180, 340, 400   # you can start from a desired frame id if needed
    while dataset.isOk():
        print('..................................')
        print('image: ', img_id)

        img = dataset.getImageColor(img_id)

        timestamp = dataset.getTimestamp()          # get current timestamp
        next_timestamp = dataset.getNextTimestamp() # get next timestamp
        frame_duration = next_timestamp-timestamp

        time_start = time.time()
        slam.track(img, img_id, timestamp)  # main SLAM function
        duration = time.time()-time_start

        # img_draw = slam.map.draw_feature_trails(img)
        # cv2.imshow('Camera', img_draw)
        # viewer3D.draw_map(slam)

        if(frame_duration > duration):
            print('sleeping for frame')
            time.sleep(frame_duration-duration)

        img_id += 1

    slam.quit()


def get_frames(idx: int = None):
    frames = []

    if idx is not None:
        # Handle single frame retrieval
        frame = {}
        frame_data = slam.map.frames[idx]
        good_indexes = [i for i, p in enumerate(frame_data.points) if p is not None and not p.is_bad]

        frame['points'] = [frame_data.points[i].pt for i in good_indexes]
        frame['kps'] = [frame_data.kps[i] for i in good_indexes]
        frame['pose'] = frame_data.Twc

        return [frame]

    else:
        # Handle retrieval of all frames
        for idx in range(len(slam.map.frames)):
            frame = get_frames(idx=idx)[0]  # Get the single frame from the list
            frames.append(frame)

    return frames


if __name__ == "__main__":
    start_slam()
