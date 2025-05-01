import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm
import OpenImageIO as oiio

gbuffers = ['albedo', 'alpha', 'depth', 'normal', 'position', 'render', 'roughness', 'viewdir']
exts =     [  '.exr',  '.exr',  '.exr',   '.exr',     '.exr',   '.exr',      '.exr',    '.exr']

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True)
args = parser.parse_args()

base = args.image_path

with open(os.path.join(base, "transforms.json")) as json_file:
    contents = json.load(json_file)

frames = contents["frames"]

for cnt in tqdm(range(len(frames))):
    frame = frames[cnt]
    dir_path, file_name = frame["file_path"].split("\\")
    i, j = int(file_name.split("_L")[0]), int(file_name.split("_L")[1])
    os.makedirs(os.path.join(base, dir_path), exist_ok=True)

    alpha_path = os.path.join(base, f"alpha_{cnt:04d}.exr")
    alpha_input_file = oiio.ImageInput.open(alpha_path)
    spec = alpha_input_file.spec()
    alpha = np.asarray(alpha_input_file.read_image(format=oiio.FLOAT)).reshape(spec.height, spec.width, spec.nchannels)
    alpha = np.where(alpha > 0.5, 1.0, 0.0).astype(np.float32)
    alpha_input_file.close()
    H, W, _ = alpha.shape
    alpha_output_file = oiio.ImageOutput.create(alpha_path)
    spec = oiio.ImageSpec(W, H, 3, oiio.FLOAT)
    alpha_output_file.open(alpha_path, spec)
    alpha_output_file.write_image(alpha.repeat(3, axis=-1))
    alpha_output_file.close()

    depth_path = os.path.join(base, f"depth_{cnt:04d}.exr")
    depth_input_file = oiio.ImageInput.open(depth_path)
    spec = depth_input_file.spec()
    depth = np.asarray(depth_input_file.read_image(format=oiio.FLOAT)).reshape(spec.height, spec.width, spec.nchannels)
    depth_input_file.close()
    H, W, _ = depth.shape
    depth_output_file = oiio.ImageOutput.create(depth_path)
    spec = oiio.ImageSpec(W, H, 3, oiio.FLOAT)
    depth_output_file.open(depth_path, spec)
    depth_output_file.write_image(depth.repeat(3, axis=-1))
    depth_output_file.close()

    position_path = os.path.join(base, f"position_{cnt:04d}.exr")
    if os.path.exists(position_path):
        position = cv2.imread(position_path, cv2.IMREAD_UNCHANGED)
        position = cv2.cvtColor(position, cv2.COLOR_BGR2RGB)

        loc = np.array(frame["transform_matrix"])[:3, 3]

        viewdir = loc - position
        viewdir /= np.linalg.norm(viewdir, axis=-1, keepdims=True)
        viewdir = viewdir * alpha
        viewdir = np.concatenate((viewdir[..., [2, 1, 0]], alpha), axis=-1).astype(np.float32)

        cv2.imwrite(os.path.join(base, f"viewdir_{cnt:04d}.exr"), viewdir)

    for gbuffer, ext in zip(gbuffers, exts):
        src_path = os.path.join(base, f"{gbuffer}_{cnt:04d}{ext}")
        dst_path = os.path.join(base, dir_path, file_name + "_" + gbuffer + ext)
        if os.path.exists(src_path):
            input = oiio.ImageInput.open(src_path)
            spec = input.spec()
            img = np.asarray(input.read_image(format=oiio.FLOAT)).reshape(spec.height, spec.width, spec.nchannels)
            input.close()
            img = img[..., :3] * alpha

            output = oiio.ImageOutput.create(dst_path)
            spec = oiio.ImageSpec(spec.width, spec.height, 3, oiio.FLOAT)
            output.open(dst_path, spec)
            output.write_image(img)
            output.close()

            os.remove(src_path)