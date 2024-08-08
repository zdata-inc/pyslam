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

if __name__ == "__main__":
    config = Config()

    dataset = dataset_factory(config.dataset_settings)

    cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                        config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                        config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                        config.DistCoef, config.cam_settings['Camera.fps'])

    num_features=2000

    tracker_type = FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn
    #tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching

    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT, CONTEXTDESC, LIGHTGLUE, XFEAT, XFEAT_XFEAT
    # WARNING: At present, SLAM is not able to support LOFTR and other "pure" image matchers (further details in the commenting notes of LOFTR in feature_tracker_configs.py).
    tracker_config = FeatureTrackerConfigs.TEST
    tracker_config['num_features'] = num_features
    tracker_config['tracker_type'] = tracker_type

    print('tracker_config: ',tracker_config)
    feature_tracker = feature_tracker_factory(**tracker_config)

    # create SLAM object 
    slam = Slam(cam, feature_tracker, None)

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

        if(frame_duration > duration):
            print('sleeping for frame')
            time.sleep(frame_duration-duration)

        img_id += 1

    slam.quit()

