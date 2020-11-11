import os
import pickle
import chumpy as ch
import numpy as np
import cv2
from psbody.mesh import Mesh
from smpl_lib.ch_smpl import Smpl
from utils.smpl_paths import SmplPaths

import global_var


class SMPL4Garment(object):
    """SMPL class for garments."""
    def __init__(self, gender):
        self.gender = gender
        smpl_model = SmplPaths(gender=gender).get_hres_smpl_model_data()
        self.smpl_base = Smpl(smpl_model)
        with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
            self.class_info = pickle.load(f)
        # self.skirt = Mesh(filename=global_var.SKIRT_TEMPLATE)

    def run(self, beta=None, theta=None, garment_d=None, garment_class=None):
        """Outputs body and garment of specified garment class given theta, beta and displacements."""
        if beta is not None:
            self.smpl_base.betas[:beta.shape[0]] = beta
        else:
            self.smpl_base.betas[:] = 0
        if theta is not None:
            self.smpl_base.pose[:] = theta
        else:
            self.smpl_base.pose[:] = 0
        self.smpl_base.v_personal[:] = 0
        if garment_d is not None and garment_class is not None:
            if 'skirt' not in garment_class:
                vert_indices = self.class_info[garment_class]['vert_indices']
                f = self.class_info[garment_class]['f']
                self.smpl_base.v_personal[vert_indices] = garment_d
                garment_m = Mesh(v=self.smpl_base.r[vert_indices], f=f)
            else:
                vert_indices = self.class_info[garment_class]['vert_indices']
                f = self.class_info[garment_class]['f']
                verts = self.smpl_base.v_poseshaped[vert_indices] + garment_d
                verts_h = ch.hstack((verts, ch.ones((verts.shape[0], 1))))
                verts = ch.sum(
                    self.smpl_base.V.T[vert_indices] * verts_h.reshape(-1, 4, 1),
                    axis=1)[:, :3]
                # if theta is not None:
                #     rotmat = self.smpl_base.A.r[:, :, 0]
                #     verts_homo = np.hstack(
                #         (verts, np.ones((verts.shape[0], 1))))
                #     verts = verts_homo.dot(rotmat.T)[:, :3]
                garment_m = Mesh(v=verts, f=f)
        else:
            garment_m = None
        self.smpl_base.v_personal[:] = 0
        body_m = Mesh(v=self.smpl_base.r, f=self.smpl_base.f)
        return body_m, garment_m
