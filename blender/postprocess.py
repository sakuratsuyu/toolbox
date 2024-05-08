import argparse
import os
import glob
import cv2
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Material:
    name: str
    color: np.ndarray
    roughness: float
    metallic: float

MATERIALS = [
    Material(
        name='Sand',
        color=np.array([0.530, 0.435, 0.248]),
        roughness=0.0,
        metallic=0.0
    ),
    Material(
        name='Brown',
        color=np.array([0.098, 0.041, 0.020]),
        roughness=0.0,
        metallic=0.0
    ),
    Material(
        name='Light_brown',
        color=np.array([0.201, 0.080, 0.034]),
        roughness=0.0,
        metallic=0.0
    ),
    Material(
        name='Black',
        color=np.array([0.003, 0.003, 0.003]),
        roughness=0.0,
        metallic=0.0
    ),
    Material(
        name='Gray',
        color=np.array([0.296, 0.323, 0.392]),
        roughness=0.0,
        metallic=0.0
    ),
    Material(
        name='Yellow',
        color=np.array([0.799, 0.503, 0.030]),
        roughness=0.0,
        metallic=0.0
    ),
    Material(
        name='tranparent',
        color=np.array([0.973, 0.973, 0.973]),
        roughness=0.141,
        metallic=0.800
    ),
    Material(
        name='RubberBand',
        color=np.array([0.042, 0.042, 0.042]),
        roughness=0.775,
        metallic=0.800
    ),
    Material(
        name='Red_Glass',
        color=np.array([1.000, 0.015, 0.003]),
        roughness=0.141,
        metallic=0.800
    )
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()

    material_index_images = glob.glob(os.path.join(args.image_path, '*', '*_material_index_output_0000.png'))
    dirs = [os.path.dirname(image) for image in material_index_images]

    for index, dir in tqdm(enumerate(dirs), total=len(dirs)):
        
        # Map material index to corresponding brdf

        image = cv2.imread(os.path.join(dir, f'{index:03d}' + '_material_index_output_0000.png'), cv2.IMREAD_UNCHANGED)[:, :, :3]

        # unique_values = np.unique(image)
        # unique_values.sort()
        # value_to_index = {value: index for index, value in enumerate(unique_values)}
        # remapped_image = np.vectorize(lambda value: value_to_index[value])(image)[:, :, 0]

        remapped_image = image[:, :, 0]

        albedo_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        roughness_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        metallic_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for (i, material) in enumerate(MATERIALS):
            mask = (remapped_image == i + 1)
            albedo_image[mask] = (material.color * 255).astype(np.uint8)
            roughness_image[mask] = int(material.roughness * 255)
            metallic_image[mask] = int(material.metallic * 255)
        
        albedo_image = cv2.cvtColor(albedo_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dir, f'{index:03d}_albedo.png'), albedo_image)
        cv2.imwrite(os.path.join(dir, f'{index:03d}_roughness.png'), roughness_image)
        cv2.imwrite(os.path.join(dir, f'{index:03d}_metallic.png'), metallic_image)

        os.remove(os.path.join(dir, f'{index:03d}_material_index_output_0000.png'))

        # Rename normal output
        os.rename(os.path.join(dir, f'{index:03d}_normal_output_0000.png'), os.path.join(dir, f'{index:03d}_normal.png'))

if __name__ == "__main__":
    main()