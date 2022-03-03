import os
import numpy as np
import cv2
import pickle
import TailorNet.global_var


def flip_theta(theta, batch=False):
    """
    flip SMPL theta along y-z plane
    if batch is True, theta shape is Nx72, otherwise 72
    """
    exg_idx = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    if batch:
        new_theta = np.reshape(theta, [-1, 24, 3])
        new_theta = new_theta[:, exg_idx]
        new_theta[:, :, 1:3] *= -1
    else:
        new_theta = np.reshape(theta, [24, 3])
        new_theta = new_theta[exg_idx]
        new_theta[:, 1:3] *= -1
    new_theta = new_theta.reshape(theta.shape)
    return new_theta


def get_Apose():
    """Return thetas for A-pose."""
    with open(os.path.join(global_var.DATA_DIR, 'apose.pkl'), 'rb') as f:
        APOSE = np.array(pickle.load(f, encoding='latin1')['pose']).astype(np.float32)
    flip_pose = flip_theta(APOSE)
    APOSE[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]] = 0
    APOSE[[14, 17, 19, 21, 23]] = flip_pose[[14, 17, 19, 21, 23]]
    APOSE = APOSE.reshape([72])
    return APOSE


def normalize_y_rotation(raw_theta):
    """Rotate along y axis so that root rotation can always face the camera.
    Theta should be a [3] or [72] numpy array.
    """
    only_global = True
    if raw_theta.shape == (72,):
        theta = raw_theta[:3]
        only_global = False
    else:
        theta = raw_theta[:]
    raw_rot = cv2.Rodrigues(theta)[0]
    rot_z = raw_rot[:, 2]
    # we should rotate along y axis counter-clockwise for t rads to make the object face the camera
    if rot_z[2] == 0:
        t = (rot_z[0] / np.abs(rot_z[0])) * np.pi / 2
    elif rot_z[2] > 0:
        t = np.arctan(rot_z[0]/rot_z[2])
    else:
        t = np.arctan(rot_z[0]/rot_z[2]) + np.pi
    cost, sint = np.cos(t), np.sin(t)
    norm_rot = np.array([[cost, 0, -sint],[0, 1, 0],[sint, 0, cost]])
    final_rot = np.matmul(norm_rot, raw_rot)
    final_theta = cv2.Rodrigues(final_rot)[0][:, 0]
    if not only_global:
        return np.concatenate([final_theta, raw_theta[3:]], 0)
    else:
        return final_theta
