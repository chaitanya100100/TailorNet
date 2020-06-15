from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
if 'SSH_CONNECTION' in os.environ:
    from utils.renderer_software import Renderer
    print("Warning: You're logged via SSH. Thus only software renderer is available, which is much slower")
else:
    import pyrender
    import trimesh
    import numpy as np

    class Renderer(object):
        """
        This is a wrapper of pyrender
        see documentation of __call__ for detailed usage
        """

        def __init__(self, img_size, bg_color=None):
            if bg_color is None:
                bg_color = np.array([0.1, 0.1, 0.1, 1.])
            self.scene = pyrender.Scene(bg_color=bg_color)
            self.focal_len = 5.
            camera = pyrender.PerspectiveCamera(yfov=np.tan(1 / self.focal_len) * 2, aspectRatio=1.0)
            camera_pose = np.eye(4, dtype=np.float32)
            self.scene.add(camera, pose=camera_pose)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0,
                                              )
            self.scene.add(light, pose=camera_pose)
            if not hasattr(img_size, '__iter__'):
                img_size = [img_size, img_size]
            self.r = pyrender.OffscreenRenderer(*img_size)

        def __call__(self, vs, fs, vcs=None, trans=(1., 0., 0.), euler=(0., 0., 0.), center=True):
            """
            This function will put the center of objects at origin point.
            vs, fs, vcs:
                vertices, faces, colors of vertices.
                They are numpy array or list of numpy array (multiple meshes)
            trans:
                It is a 3 element tuple. The first is scale factor. The last two is x,y translation
            euler:
                euler angle of objects (degree not radian). It follows the order of YXZ,
                which means Y-axis, X-axis, Z-axis are yaw, pitch, roll respectively.
            """
            if isinstance(vs, np.ndarray):
                vs = [vs]
                fs = [fs]
                vcs = [vcs]
            ms = []
            mnodes = []
            vss = np.concatenate(vs, 0)
            cen = (np.max(vss, 0, keepdims=True) + np.min(vss, 0, keepdims=True)) / 2.
            rotmat = self.euler2rotmat(euler)
            for v, f, vs in zip(vs, fs, vcs):
                trans_v = v - cen if center else v
                trans_v = np.einsum('pq,nq->np', rotmat, trans_v)
                trans_v[:, :2] += np.expand_dims(np.array(trans[1:]), 0)
                trans_v[:, 2] -= self.focal_len / trans[0]
                ms.append(trimesh.Trimesh(vertices=trans_v, faces=f, vertex_colors=vs))
            for m in ms:
                mnode = self.scene.add(pyrender.Mesh.from_trimesh(m))
                mnodes.append(mnode)
            img, depth = self.r.render(self.scene)
            for mnode in mnodes:
                self.scene.remove_node(mnode)
            return img

        @staticmethod
        def euler2rotmat(euler):
            euler = np.array(euler)*np.pi/180.
            se, ce = np.sin(euler), np.cos(euler)
            s1, c1 = se[0], ce[0]
            s2, c2 = se[1], ce[1]
            s3, c3 = se[2], ce[2]
            return np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                             [c2*s3, c2*c3, -s2],
                             [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])

if __name__ == '__main__':
    from psbody.mesh import Mesh
    import cv2
    import numpy as np
    import global_var

    m1 = Mesh(filename='/home/zliao/cloth-anim/work/data/md/cloth_test/121611457711203/apose_avatar.obj')
    m2 = Mesh(filename='/home/zliao/cloth-anim/work/data/md/cloth_test/121611457711203/result_Pants.obj')
    vs = [m1.v, m2.v]
    fs = [m1.f, m2.f]
    colors = [np.array([0.6, 0.6, 0.9]), np.array([0.8, 0.5, 0.3])]
    renderer = Renderer(800)
    img = renderer(vs, fs, colors)
    trans_img = renderer(vs, fs, colors, trans=(0.8, 0.3, 0.3))
    euler_img = renderer(vs, fs, colors, euler=(45, 20, 0))
    img = np.concatenate([img, trans_img, euler_img], 1)
    cv2.imwrite(os.path.join(global_var.DATA_DIR, 'img.png'), img)
