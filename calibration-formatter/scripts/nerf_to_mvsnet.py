import sys
sys.path.append("../")

import os
import json
import argparse
import shutil
import numpy as np
import cv2

from utils.coordinates_utils import opengl_to_opencv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='path to the data root directory, containing transforms_train.json')
    parser.add_argument('--file_name', type=str, default='transforms_train.json')
    parser.add_argument('--convert_format', action='store_true', help='convert images to jpg format')
    args = parser.parse_args()

    # read
    with open(os.path.join(args.data_root, args.file_name), 'r') as f:
        transforms = json.load(f)

    H, W = 800, 800
    if 'h' in transforms and 'w' in transforms:
        H, W = transforms["h"], transforms["w"]

    intrinsic = []
    extrinsic = []
    for _, frame in enumerate(transforms["frames"]):

        # extri
        c2w_opengl = np.array(frame["transform_matrix"]).astype(np.float32)
        # w2c_opengl = np.linalg.inv(c2w_opengl)
        c2w_opencv = opengl_to_opencv(c2w_opengl)
        w2c_opencv = np.linalg.inv(c2w_opencv)

        # intri
        if 'fl_x' in frame:
            fx, fy = frame["fl_x"], frame["fl_y"]
            cx, cy = frame["cx"], frame["cy"]
            H, W = frame["h"], frame["w"]
        else:
            if H is None or W is None:
                raise ValueError('Height and width must be provided in the transforms file, or should be specified in the script.')
            fx, fy = 0.5 * W / np.tan(0.5 * transforms["camera_angle_x"]), 0.5 * W / np.tan(0.5 * transforms["camera_angle_x"])
            cx, cy = 0.5 * W, 0.5 * H

        intrinsic.append(np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ]))
        # extrinsic.append(w2c_opengl)
        extrinsic.append(w2c_opencv)


    # write
    cam_dir = os.path.join(args.data_root, 'cams')
    try:
        os.makedirs(cam_dir)
    except os.error:
        print(cam_dir + ' already exist.')

    for i, _ in enumerate(transforms["frames"]):
        with open(os.path.join(cam_dir, '%08d_cam.txt' % i), 'w') as f:
            f.write('extrinsic\n')
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[i][j, k]) + ' ')
                f.write('\n')
            f.write('\nintrinsic\n')
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic[i][j, k]) + ' ')
                f.write('\n')
            # f.write('\n%f %f %f %f\n' % (depth_ranges[i+1][0], depth_ranges[i+1][1], depth_ranges[i+1][2], depth_ranges[i+1][3]))

    # copy images
    img_dir = os.path.join(args.data_root, 'images')
    try:
        os.makedirs(img_dir)
    except os.error:
        print(img_dir + ' already exist.')
    for i, frame in enumerate(transforms["frames"]):
        if args.convert_format:
            img = cv2.imread(os.path.join(args.data_root, frame["file_path"] + ".png"))
            cv2.imwrite(os.path.join(img_dir, '%08d.jpg' % i), img)
        else:
            shutil.copyfile(os.path.join(args.data_root, frame["file_path"] + ".png"), os.path.join(img_dir, '%08d.jpg' % i))


if __name__ == '__main__':
    main()