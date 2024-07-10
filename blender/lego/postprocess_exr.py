import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm

gbuffers = ['albedo', 'alpha', 'normal', 'position', 'render', 'roughness', 'viewdir']
exts = ['.exr', '.png', '.exr', '.exr', '.exr', '.exr', '.exr', '.exr']

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True)
args = parser.parse_args()

base = args.image_path

with open(os.path.join(base, "transforms.json")) as json_file:
    contents = json.load(json_file)

frames = contents["frames"]

for i in tqdm(range(len(frames))):
    path = os.path.join(base, f"alpha_{i:04d}.png")
    alpha = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., 0] / 255.0

    path = os.path.join(base, f"position_{i:04d}.exr")
    position = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    position = cv2.cvtColor(position, cv2.COLOR_BGR2RGB)

    frame = frames[i]
    loc = np.array(frame["transform_matrix"])[:3, 3]

    viewdir = loc - position
    viewdir /= np.linalg.norm(viewdir, axis=-1, keepdims=True)
    viewdir = viewdir * alpha[..., None]
    viewdir = np.concatenate((viewdir[..., [2, 1, 0]], alpha[..., None]), axis=-1).astype(np.float32)

    cv2.imwrite(os.path.join(base, f"viewdir_{i:04d}.exr"), viewdir)

    for gbuffer, ext in zip(gbuffers, exts):
        os.rename(os.path.join(base, f"{gbuffer}_{i:04d}{ext}"), os.path.join(base, f"{i:03d}", f"{i:03d}_{gbuffer}{ext}"))