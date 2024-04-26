import numpy as np

def opengl_to_opencv(c2w_opengl):
    """
    Convert OpenGL camera to OpenCV camera.
    :param c2w_opengl: 4x4 matrix, OpenGL camera-to-world matrix.
    :return: 4x4 matrix, OpenCV camera-to-world matrix.
    """

    transform_matrix = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])

    c2w_opencv = c2w_opengl @ transform_matrix

    return c2w_opencv

def opencv_to_opengl(c2w_opencv):
    """
    Convert OpenCV camera to OpenGL camera. Actually the same as `opengl_to_opencv`.
    :param c2w_opencv: 4x4 matrix, OpenCV camera-to-world matrix.
    :return: 4x4 matrix, OpenGL camera-to-world matrix.
    """

    transform_matrix = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])

    c2w_opengl = c2w_opencv @ transform_matrix

    return c2w_opengl