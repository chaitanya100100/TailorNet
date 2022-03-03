import numpy as np
import scipy
import scipy.sparse as sp
from TailorNet.psbody.mesh import Mesh, MeshViewer


def numpy_laplacian_uniform(v, f):
    """Computes uniform laplacian operator on mesh."""
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize
    from psbody.mesh.topology.connectivity import get_vert_connectivity

    connectivity = get_vert_connectivity(Mesh(v=v, f=f))
    # connectivity is a sparse matrix, and np.clip can not applied directly on
    # a sparse matrix.
    connectivity.data = np.clip(connectivity.data, 0, 1)
    lap = normalize(connectivity, norm='l1', axis=1)
    lap = lap - sp.eye(connectivity.shape[0])

    return lap


def numpy_laplacian_cot(v, f):
    """Computes cotangent laplacian operator on mesh."""
    n = len(v)

    v_a = f[:, 0]
    v_b = f[:, 1]
    v_c = f[:, 2]

    ab = v[v_a] - v[v_b]
    bc = v[v_b] - v[v_c]
    ca = v[v_c] - v[v_a]

    cot_a = -1 * (ab * ca).sum(axis=1) / (np.sqrt(np.sum(np.cross(ab, ca) ** 2, axis=-1)) + 1.e-10)
    cot_b = -1 * (bc * ab).sum(axis=1) / (np.sqrt(np.sum(np.cross(bc, ab) ** 2, axis=-1)) + 1.e-10)
    cot_c = -1 * (ca * bc).sum(axis=1) / (np.sqrt(np.sum(np.cross(ca, bc) ** 2, axis=-1)) + 1.e-10)

    I = np.concatenate((v_a, v_c, v_a, v_b, v_b, v_c))
    J = np.concatenate((v_c, v_a, v_b, v_a, v_c, v_b))
    W = 0.5 * np.concatenate((cot_b, cot_b, cot_c, cot_c, cot_a, cot_a))

    L = sp.csr_matrix((W, (I, J)), shape=(n, n))
    L = L - sp.spdiags(L * np.ones(n), 0, n, n)

    return L


def direct_smoothing(v, f, smoothness=0.1, Ltype='cotangent'):
    """Apply direct smoothing on mesh."""
    if Ltype == 'cotangent':
        L = numpy_laplacian_cot(v, f)
    elif Ltype == 'uniform':
        L = numpy_laplacian_uniform(v, f)
    else:
        raise AttributeError
    new_v = v + smoothness * L.dot(v)
    return new_v


class DiffusionSmoothing(object):
    """A class useful to apply smoothing repeatedly in efficient manner on the same-topology meshes."""

    def __init__(self, v, f):
        """Computes and stores necessary variables.

        v is only used for getting total number of vertices. f defines the topology.
        """
        self.num_v = v.shape[0]
        self.v = v
        self.f = f
        self.set_boundary_ids_and_mats(v, f)
        self.uniL = None

    def get_uniform_lap_smoothing(self):
        """Computes uniform laplacian for smoothing.

        Boundary vertices are smoothed not by all neighbors but only neighboring
        boundary vertices in order to prevent boundary shrinking.
        """
        L = numpy_laplacian_uniform(self.v, self.f)

        # remove rows corresponding to boundary vertices
        for row in self.b_ids:
            L.data[L.indptr[row]:L.indptr[row + 1]] = 0
        L.eliminate_zeros()

        num_b = self.b_ids.shape[0]
        I = np.tile(self.b_ids, 3)
        J = np.hstack((
            self.b_ids,
            self.b_ids[self.l_ids],
            self.b_ids[self.r_ids],
        ))
        W = np.hstack((
            -1 * np.ones(num_b),
            0.5 * np.ones(num_b),
            0.5 * np.ones(num_b),
        ))
        mat = sp.csr_matrix((W, (I, J)), shape=(self.num_v, self.num_v))
        L = L + mat
        return L

    def set_boundary_ids_and_mats(self, v, f):
        from .geometry import get_boundary_verts
        _, b_rings = get_boundary_verts(v, f)

        def shift_left(ls, k):
            return ls[k:] + ls[:k]

        b_ids = []
        l_ids = []
        r_ids = []
        for rg in b_rings:
            tmp = list(range(len(b_ids), len(b_ids) + len(rg)))
            ltmp = shift_left(tmp, 1)
            rtmp = shift_left(tmp, -1)
            l_ids.extend(ltmp)
            r_ids.extend(rtmp)

            b_ids.extend(rg)

        b_ids = np.asarray(b_ids, dtype=np.int64)
        num_b = b_ids.shape[0]
        m_ids = np.arange(num_b, dtype=np.int64)
        l_ids = np.asarray(l_ids, dtype=np.int64)
        r_ids = np.asarray(r_ids, dtype=np.int64)

        self.right_edge_mat = sp.csr_matrix((
            np.hstack((-1*np.ones(num_b), np.ones(num_b))),
            (np.hstack((m_ids, m_ids)), np.hstack((m_ids, r_ids)))
        ), shape=(num_b, num_b)
        )

        self.left_edge_mat = sp.csr_matrix((
            np.hstack((-1 * np.ones(num_b), np.ones(num_b))),
            (np.hstack((m_ids, m_ids)), np.hstack((m_ids, l_ids)))
        ), shape=(num_b, num_b)
        )

        # boundary vertex ids
        self.b_ids = b_ids
        # left and right boundary neighbour vertex ids
        self.l_ids = l_ids
        self.r_ids = r_ids

    def smooth_cotlap(self, verts, smoothness=0.03):
        """Smooth using cotangent laplacian.

        Boundary vertices are smoothed only by neighboring boundary vertices
        in order to prevent boundary shrinking.
        """
        L = numpy_laplacian_cot(verts, self.f)
        new_verts = verts + smoothness * L.dot(verts)

        b_verts = verts[self.b_ids]
        le = 1. / (np.linalg.norm(self.left_edge_mat.dot(b_verts), axis=-1) + 1.0e-10)
        ri = 1. / (np.linalg.norm(self.right_edge_mat.dot(b_verts), axis=-1) + 1.0e-10)

        num_b = b_verts.shape[0]
        I = np.tile(np.arange(num_b), 3)
        J = np.hstack((
            np.arange(num_b),
            self.l_ids,
            self.r_ids,
        ))
        W = np.hstack((
            -1*np.ones(num_b),
            le / (le + ri),
            ri / (le + ri),
        ))
        mat = sp.csr_matrix((W, (I, J)), shape=(num_b, num_b))
        new_verts[self.b_ids] = verts[self.b_ids] + smoothness * mat.dot(verts[self.b_ids])
        return new_verts

    def smooth_uniform(self, verts, smoothness=0.03):
        """Smooth using uniform laplacian.

        Boundary vertices are smoothed only by neighboring boundary vertices
        in order to prevent boundary shrinking.
        """
        if self.uniL is None:
            self.uniL = self.get_uniform_lap_smoothing()
        new_verts = verts + smoothness * self.uniL.dot(verts)
        return new_verts

    def smooth(self, verts, smoothness=0.03, n=1, Ltype="cotangent"):
        assert(Ltype in ["cotangent", "uniform"])
        for i in range(n):
            if Ltype == 'uniform':
                verts = self.smooth_uniform(verts, smoothness)
            else:
                verts = self.smooth_cotlap(verts, smoothness)
        return verts


if __name__ == "__main__":
    IS_SMPL = True
    fpath = "/BS/cpatel/work/data/learn_anim/mixture_exp31/000_0/smooth_TShirtNoCoat/0990/pred_0.ply"

    if not IS_SMPL:
        ms = Mesh(filename=fpath)
    else:
        from utils.smpl_paths import SmplPaths

        dp = SmplPaths(gender='female')
        smpl = dp.get_smpl()
        ms = Mesh(v=smpl.r, f=smpl.f)

    smoothing = DiffusionSmoothing(ms.v, ms.f)

    verts_smooth = ms.v.copy()
    for i in range(20):
        verts_smooth = smoothing.smooth(verts_smooth, smoothness=0.05)
    ms_smooth = Mesh(v=verts_smooth, f=ms.f)

    # from psbody.mesh import MeshViewers
    # mvs = MeshViewers((1,3))
    # mvs[0][0].set_static_meshes([ms])
    # mvs[0][1].set_static_meshes([ms_smooth])
    # mvs[0][2].set_static_meshes([ms_smooth2])
    # import ipdb
    # ipdb.set_trace()

    ms.write_ply("/BS/cpatel/work/orig.ply")
    ms_smooth.write_ply("/BS/cpatel/work/smooth.ply")
