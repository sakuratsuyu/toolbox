import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import math
from typing import List, Tuple

rotations: List[np.array] = [
    np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0]
    ]), # look at +y (+x in camera coordinate)
    np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ]), # look at -y (-x in camera coordinate)
    np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0]
    ]), # look at +z (+y in camera coordinate)
    np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ]), # look at -z (-y in camera coordinate)
    np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]), # look at -x (-z in camera coordinate)
    np.array([
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]), # look at +x (+z in camera coordinate)
]

def get_envmap_dirs(res: List[int] = [256, 512]) -> Tuple[np.array, np.array]:
    gy, gx = np.meshgrid(
        np.linspace(0.0, 1.0 - 1.0 / res[0], res[0]),
        np.linspace(-1.0, 1.0 - 1.0 / res[1], res[1]),
        indexing="ij",
    )
    d_theta, d_phi = np.pi / res[0], 2 * np.pi / res[1]

    sintheta, costheta = np.sin(gy * np.pi), np.cos(gy * np.pi)
    sinphi, cosphi = np.sin(gx * np.pi), np.cos(gx * np.pi)

    envmap_dirs = np.stack((sintheta * sinphi, costheta, sintheta * cosphi), axis=-1)  # [H, W, 3]

    solid_angles = (sintheta * d_theta * d_phi)[..., None]  # [H, W, 1]
    print(f"solid_angles_sum error: {solid_angles.sum() - 4 * np.pi}")

    return solid_angles, envmap_dirs

def direction_to_cubemap_uv(directions):
    """Given (N, 3) directions, return face_idx and (u, v) coords inside that face"""

    N, _ = directions.shape
    abs_directions = np.abs(directions)
    max_axis = np.argmax(abs_directions, axis=-1)

    uc = np.zeros(N)
    vc = np.zeros(N)
    face = np.zeros(N)

    # +X face
    mask = (max_axis == 0) & (directions[:, 0] > 0)
    uc[mask] = -directions[mask, 2] / abs_directions[mask, 0]
    vc[mask] = -directions[mask, 1] / abs_directions[mask, 0]
    face[mask] = 0
    # -X face
    mask = (max_axis == 0) & (directions[:, 0] < 0)
    uc[mask] = directions[mask, 2] / abs_directions[mask, 0]
    vc[mask] = -directions[mask, 1] / abs_directions[mask, 0]
    face[mask] = 1
    # +Y face
    mask = (max_axis == 1) & (directions[:, 1] > 0)
    uc[mask] = directions[mask, 0] / abs_directions[mask, 1]
    vc[mask] = directions[mask, 2] / abs_directions[mask, 1]
    face[mask] = 2
    # -Y face
    mask = (max_axis == 1) & (directions[:, 1] < 0)
    uc[mask] = directions[mask, 0] / abs_directions[mask, 1]
    vc[mask] = -directions[mask, 2] / abs_directions[mask, 1]
    face[mask] = 3
    # +Z face
    mask = (max_axis == 2) & (directions[:, 2] > 0)
    uc[mask] = directions[mask, 0] / abs_directions[mask, 2]
    vc[mask] = -directions[mask, 1] / abs_directions[mask, 2]
    face[mask] = 4
    # -Z face
    mask = (max_axis == 2) & (directions[:, 2] < 0)
    uc[mask] = -directions[mask, 0] / abs_directions[mask, 2]
    vc[mask] = -directions[mask, 1] / abs_directions[mask, 2]
    face[mask] = 5

    u = (uc + 1) / 2
    v = (vc + 1) / 2
    return face, u, v

def sample_cubemap(cubemaps, face_idx, u, v):
    """Sample colors from cubemap faces"""

    H, W, _ = cubemaps[0].shape
    u_pix = np.clip((u * (W - 1)).astype(np.int32), 0, W - 1)
    v_pix = np.clip((v * (H - 1)).astype(np.int32), 0, H - 1)
    color = np.zeros((len(face_idx), 3), dtype=np.float32)
    for i in range(6):
        mask = (face_idx == i)
        color[mask] = cubemaps[i][v_pix[mask], u_pix[mask]]
    return color

def build_envmap(cubemaps, envmap_dirs):
    env_h, env_w, _ = envmap_dirs.shape
    face_idx, uu, vv = direction_to_cubemap_uv(envmap_dirs.reshape(-1, 3))
    color = sample_cubemap(cubemaps, face_idx, uu, vv)
    envmap = color.reshape((env_h, env_w, 3))

    return envmap

def get_projection_matrix(znear: float, zfar: float, fovX: float, fovY: float) -> np.array:
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def transform_and_project(vertices, w2c, proj):

    ones = np.ones_like(vertices[:, :1])
    vertices_homo = np.concatenate([vertices, ones], axis=-1)
    vertices_homo_camera = np.einsum('ij, kj->ki', w2c, vertices_homo)
    vertices_clip = np.einsum('ij, kj->ki', proj, vertices_homo_camera)

    return vertices_clip

def ndc_clip_polygon(vertices_clip):

    # Sutherland–Hodgman clip
    # def clip_against_plane(vertices, plane):
    #     clipped = []
    #     prev = vertices[-1]
    #     prev_distance = np.dot(plane, prev)

    #     for curr in vertices:
    #         curr_distance = np.dot(plane, curr)
    #         if curr_distance * prev_distance < 0:
    #             d = prev_distance / (prev_distance - curr_distance)
    #             intersect = prev + d * (curr - prev)
    #             clipped.append(intersect)
    #         if curr_distance >= 0:
    #             clipped.append(curr)
    #         prev = curr
    #         prev_distance = curr_distance

    #     return clipped

    def clip_against_plane(vertices, plane):

        curr = vertices
        prev = np.roll(vertices, shift=1, axis=0)

        curr_distance = np.sum(curr * plane[None, :], axis=1)
        prev_distance = np.sum(prev * plane[None, :], axis=1)

        intersect_mask = curr_distance * prev_distance < 0
        d = prev_distance / (prev_distance - curr_distance)
        intersect = prev + d[:, None] * (curr - prev)
        
        keep_mask = curr_distance >= 0

        indices = np.concatenate([
            np.nonzero(intersect_mask)[0] * 2,
            np.nonzero(keep_mask)[0] * 2 + 1
        ], axis=0)
        vertices_clipped = np.concatenate([intersect[intersect_mask], curr[keep_mask]], axis=0)
        
        sorted_indices = np.argsort(indices)
        vertices_clipped = vertices_clipped[sorted_indices]
        return vertices_clipped

    clip_planes = [
        np.array([ 1.0,  0.0,  0.0, 1.0]),  # 左平面 x >= -w
        np.array([-1.0,  0.0,  0.0, 1.0]),  # 右平面 x <= w
        np.array([ 0.0,  1.0,  0.0, 1.0]),  # 下平面 y >= -w
        np.array([ 0.0, -1.0,  0.0, 1.0]),  # 上平面 y <= w
        np.array([ 0.0,  0.0,  1.0, 1.0]),  # 近平面 z >= -w
        np.array([ 0.0,  0.0, -1.0, 1.0])   # 远平面 z <= w
    ]

    vertices = np.copy(vertices_clip)
    for plane in clip_planes:
        vertices = clip_against_plane(vertices, plane)
        if len(vertices) == 0:
            break
        # vertices = np.stack(vertices)

    return vertices

def draw_polygon_ndc(ndc_vertices, image_size):
    img = np.zeros((image_size, image_size, 3), dtype=np.float32)

    if len(ndc_vertices) == 0:
        return img

    ndc = ndc_vertices[:, :3] / ndc_vertices[:, 3:4]
    screen_xy = (ndc[:, :2] * 0.5 + 0.5) * image_size
    screen_xy = screen_xy.astype(np.int32)
    
    cv2.fillPoly(img, [screen_xy], (1.0, 1.0, 1.0))

    return img

def main():
    solid_angles, envmap_dirs = get_envmap_dirs() # [H, W, 1], [H, W, 3]

    vertices = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0]
    ])

    position = np.array([0.0, 0.0, -0.5])

    imgs = []
    for rotation in rotations:
        c2w = np.eye(4)
        c2w = np.zeros((4, 4))
        c2w[:3, :3] = rotation
        c2w[:3, 3] = position
        c2w[3, 3] = 1.0
        c2w[:3, 1:3] *= -1 # opengl to opencv coordinate
        w2c = np.linalg.inv(c2w)

        proj = get_projection_matrix(znear=0.01, zfar=100, fovX=math.pi * 0.5, fovY=math.pi * 0.5)

        vertices_clip = transform_and_project(vertices, w2c, proj)

        vertices_clipped = ndc_clip_polygon(vertices_clip)

        img = draw_polygon_ndc(vertices_clipped, image_size=256)
        img = np.array(img)
        imgs.append(img)
    
    envmap = build_envmap(np.stack(imgs), envmap_dirs)

    cv2.imwrite("envmap.exr", cv2.cvtColor(envmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite("cube.exr", cv2.cvtColor(np.concat(imgs, axis=1), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
