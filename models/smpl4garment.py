import os
import pickle
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
        if garment_d is not None and garment_class is not None:
            if 'skirt' not in garment_class:
                vert_indices = self.class_info[garment_class]['vert_indices']
                f = self.class_info[garment_class]['f']
                self.smpl_base.v_personal[vert_indices] = garment_d
                garment_m = Mesh(v=self.smpl_base.r[vert_indices], f=f)
            else:
                verts = garment_d
                if theta is not None:
                    rotmat = cv2.Rodrigues(theta[:3])[0]
                    verts = verts.dot(rotmat.T)
                # verts += self.smpl_base.J.r[:1, :]
                f = self.class_info[garment_class]['f']
                garment_m = Mesh(v=verts, f=f)
        else:
            garment_m = None
        self.smpl_base.v_personal[:] = 0
        body_m = Mesh(v=self.smpl_base.r, f=self.smpl_base.f)
        return body_m, garment_m
