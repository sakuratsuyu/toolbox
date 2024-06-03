import argparse
import os
import glob
import cv2
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()

    dirs = os.listdir(args.image_path)
    dirs.sort()
    postfix = '0002'

    for index, dir in tqdm(enumerate(dirs), total=len(dirs)):

        if dir == 'transforms.json':
            continue

        index = int(dir[:3])

        # Rename normal output
        if os.path.exists(os.path.join(args.image_path, dir, f'{index:03d}_normal_output_{postfix}.png')):
            os.rename(os.path.join(args.image_path, dir, f'{index:03d}_alpha_output_{postfix}.png'), os.path.join(args.image_path, dir, f'{index:03d}_alpha.png'))
            os.rename(os.path.join(args.image_path, dir, f'{index:03d}_normal_output_{postfix}.png'), os.path.join(args.image_path, dir, f'{index:03d}_normal.png'))
            os.rename(os.path.join(args.image_path, dir, f'{index:03d}_diffuse_output_{postfix}.png'), os.path.join(args.image_path, dir, f'{index:03d}_diffuse.png'))
            os.rename(os.path.join(args.image_path, dir, f'{index:03d}_roughness_output_{postfix}.png'), os.path.join(args.image_path, dir, f'{index:03d}_roughness.png'))

if __name__ == "__main__":
    main()