import numpy as np

import scipy.sparse as sp
from scipy.sparse import vstack, csr_matrix
from scipy.sparse.linalg import spsolve
from psbody.mesh import Mesh

from psbody.mesh import Mesh
from psbody.mesh.geometry.vert_normals import VertNormals
from psbody.mesh.geometry.tri_normals import TriNormals
from psbody.mesh.search import AabbTree
from utils.diffusion_smoothing import numpy_laplacian_uniform as laplacian


def get_nearest_points_and_normals(vert, base_verts, base_faces):
    """For each vertex of `vert`, find nearest surface points on
    base mesh (`base_verts`, `base_faces`).
    """
    fn = TriNormals(v=base_verts, f=base_faces).reshape((-1, 3))
    vn = VertNormals(v=base_verts, f=base_faces).reshape((-1, 3))

    tree = AabbTree(Mesh(v=base_verts, f=base_faces))
    nearest_tri, nearest_part, nearest_point = tree.nearest(vert, nearest_part=True)
    nearest_tri = nearest_tri.ravel().astype(np.long)
    nearest_part = nearest_part.ravel().astype(np.long)

    nearest_normals = np.zeros_like(vert)

    # nearest_part tells you whether the closest point in triangle abc is
    # in the interior (0), on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)
    cl_tri_idxs = np.nonzero(nearest_part == 0)[0].astype(np.int)
    cl_vrt_idxs = np.nonzero(nearest_part > 3)[0].astype(np.int)
    cl_edg_idxs = np.nonzero((nearest_part <= 3) & (nearest_part > 0))[0].astype(np.int)

    nt = nearest_tri[cl_tri_idxs]
    nearest_normals[cl_tri_idxs] = fn[nt]

    nt = nearest_tri[cl_vrt_idxs]
    npp = nearest_part[cl_vrt_idxs] - 4
    nearest_normals[cl_vrt_idxs] = vn[base_faces[nt, npp]]

    nt = nearest_tri[cl_edg_idxs]
    npp = nearest_part[cl_edg_idxs] - 1
    nearest_normals[cl_edg_idxs] += vn[base_faces[nt, npp]]
    npp = np.mod(nearest_part[cl_edg_idxs], 3)
    nearest_normals[cl_edg_idxs] += vn[base_faces[nt, npp]]

    nearest_normals = nearest_normals / (np.linalg.norm(nearest_normals, axis=-1, keepdims=True) + 1.e-10)

    return nearest_point, nearest_normals


def remove_interpenetration_fast(mesh, base, L=None):
    """Deforms `mesh` to remove its interpenetration from `base`.
    This is posed as least square optimization problem which can be solved
    faster with sparse solver.
    """

    eps = 0.001
    ww = 2.0
    nverts = mesh.v.shape[0]

    if L is None:
        L = laplacian(mesh.v, mesh.f)

    nearest_points, nearest_normals = get_nearest_points_and_normals(mesh.v, base.v, base.f)
    direction = np.sign( np.sum((mesh.v - nearest_points) * nearest_normals, axis=-1) )

    indices = np.where(direction < 0)[0]

    pentgt_points = nearest_points[indices] - mesh.v[indices]
    pentgt_points = nearest_points[indices] \
                    + eps * pentgt_points / np.expand_dims(0.0001 + np.linalg.norm(pentgt_points, axis=1), 1)
    tgt_points = mesh.v.copy()
    tgt_points[indices] = ww * pentgt_points

    rc = np.arange(nverts)
    data = np.ones(nverts)
    data[indices] *= ww
    I = csr_matrix((data, (rc, rc)), shape=(nverts, nverts))

    A = vstack([L, I])
    b = np.vstack((
        L.dot(mesh.v),
        tgt_points
    ))

    res = spsolve(A.T.dot(A), A.T.dot(b))
    mres = Mesh(v=res, f=mesh.f)
    return mres


if __name__ == '__main__':
    import os
    ROOT = "/BS/cpatel/work/data/learn_anim/mixture_exp31/000_0/smooth_TShirtNoCoat/0990/"
    body = Mesh(filename=os.path.join(ROOT, "body_160.ply"))
    mesh = Mesh(filename=os.path.join(ROOT, "pred_160.ply"))

    mesh1 = remove_interpenetration_fast(mesh, body)
    mesh1.write_ply("/BS/cpatel/work/proccessed.ply")
    mesh.write_ply("/BS/cpatel/work/orig.ply")
    body.write_ply("/BS/cpatel/work/body.ply")

    # from psbody.mesh import MeshViewers
    # mvs = MeshViewers((1, 2))
    # mesh1.set_vertex_colors_from_weights(np.linalg.norm(mesh.v - mesh1.v, axis=1))
    # mesh.set_vertex_colors_from_weights(np.linalg.norm(mesh.v - mesh1.v, axis=1))
    # # mesh1.set_vertex_colors_from_weights(np.zeros(mesh.v.shape[0]))
    # # mesh.set_vertex_colors_from_weights(np.zeros(mesh.v.shape[0]))
    # mvs[0][0].set_static_meshes([mesh, body])
    # mvs[0][1].set_static_meshes([mesh1, body])
    # mesh1.show()

    import ipdb
    ipdb.set_trace()