# from torchvision import transforms as T
# import torch
# import torchvision
import cv2 as cv
import numpy as np
import random

import carla
from utils import cv_from_carla_image, cv_from_semantic_image, cv_from_depth_image


class AutoPilot:
    def __init__(self, show_data=True, save_data=False):
        self.sgbm = cv.StereoSGBM_create(
            numDisparities=144,
            blockSize=3,
            P1=126, P2=1024,
            mode=cv.StereoSGBM_MODE_SGBM_3WAY)

        self.save_data = False
        self.show_data = False

    def __call__(self, data):
        snapshot, left_image, right_image, semantic_gt, depth_gt = data
        print('Timestamp: {}'.format(snapshot.timestamp.platform_timestamp))

        left = cv_from_carla_image(left_image)
        right = cv_from_carla_image(right_image)
        semantic = cv_from_semantic_image(semantic)
        gt_depth = cv_from_carla_image(depth)
        depth = cv_from_depth_image(depth)

        if self.save_data:
            create_name = lambda x: str(snapshot.timestamp.platform_timestamp)+str(x)+'.png'
            cv.imwrite(create_name('left'), left)
            cv.imwrite(create_name('right'), right)
            cv.imwrite(create_name('depth'), gt_depth)

        disp = self.sgbm.compute(left, right) / 16.0
        disp = disp.astype(np.uint8)
        disp[np.where(disp < 1)] = 1


        if self.show_data:
            cv.imshow('disp', cv.applyColorMap(disp, cv.COLORMAP_JET))

            road_mask = cv.inRange(semantic, 6, 7)
            cv.imshow('road mask', road_mask)

            cv.imshow('left', left)
            cv.waitKey(10)

        return carla.VehicleControl(
            steer=random.random()*0.1-0.05,
            throttle=0.5,
            gear=1)
