import numpy as np
import chumpy as ch
from TailorNet.psbody.mesh import Mesh
import torch
import scipy.sparse as sp
from chumpy.utils import row, col


def get_face_normals(verts, faces):
    num_batch = verts.size(0)
    num_faces = faces.size(0)

    # faces by vertices
    fbv = torch.index_select(verts, 1, faces.view(-1)).view(num_batch, num_faces, 3, 3)
    normals = torch.cross(fbv[:, :, 1] - fbv[:, :, 0], fbv[:, :, 2] - fbv[:, :, 0], dim=2)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1.e-10)
    return normals


def get_vertex_normals(verts, faces, ret_face_normals=False):
    num_faces = faces.size(0)
    num_verts = verts.size(1)
    face_normals = get_face_normals(verts, faces)

    FID = torch.arange(num_faces).unsqueeze(1).repeat(1, 3).view(-1)
    VID = faces.view(-1)
    data = torch.ones_like(FID, dtype=torch.float32)

    mat = torch.sparse_coo_tensor(
        indices=torch.stack((VID, FID)),
        values=data,
        size=(num_verts, num_faces)
    )
    degree = torch.sparse.sum(mat, dim=1).to_dense()
    vertex_normals = torch.stack((
        torch.sparse.mm(mat, face_normals[:, :, 0].t()),
        torch.sparse.mm(mat, face_normals[:, :, 1].t()),
        torch.sparse.mm(mat, face_normals[:, :, 2].t()),
    ), dim=-1)
    vertex_normals = vertex_normals.transpose(1, 0) / degree.unsqueeze(0).unsqueeze(-1)
    vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=-1, keepdim=True) + 1.e-10)

    if ret_face_normals:
        return vertex_normals, face_normals
    else:
        return vertex_normals


def unpose_garment(smpl, v_free, vert_indices=None):
    smpl.v_personal[:] = 0
    c = smpl[vert_indices]
    E = {
        'v_personal_high': c - v_free
    }
    ch.minimize(E, x0=[smpl.v_personal], options={'e_3': .00001})
    smpl.pose[:] = 0
    smpl.trans[:] = 0

    return Mesh(smpl.r, smpl.f).keep_vertices(vert_indices), np.copy(np.array(smpl.v_personal))


def merge_mesh(vs, fs, vcs):
    v_num = 0
    new_fs = [fs[0]]
    new_vcs = []
    for i in range(len(vs)):
        if i >= 1:
            v_num += vs[i-1].shape[0]
            new_fs.append(fs[i]+v_num)
        if vcs is not None:
            if vcs[i].ndim == 1:
                new_vcs.append(np.tile(np.expand_dims(vcs[i], 0), [vs[i].shape[0], 1]))
            else:
                new_vcs.append(vcs)
    vs = np.concatenate(vs, 0)
    new_fs = np.concatenate(new_fs, 0)
    if vcs is not None:
        vcs = np.concatenate(new_vcs, 0)
    return vs, new_fs, vcs


def get_edges2face(faces):
    from itertools import combinations
    from collections import OrderedDict
    # Returns a structure that contains the faces corresponding to every edge
    edges = OrderedDict()
    for iface, f in enumerate(faces):
        sorted_face_edges = tuple(combinations(sorted(f), 2))
        for sorted_face_edge in sorted_face_edges:
            if sorted_face_edge in edges:
                edges[sorted_face_edge].faces.add(iface)
            else:
                edges[sorted_face_edge] = lambda:0
                edges[sorted_face_edge].faces = set([iface])
    return edges


def get_boundary_verts(verts, faces, connected_boundaries=True, connected_faces=False):
    """
     Given a mesh returns boundary vertices
     if connected_boundaries is True it returs a list of lists
     OUTPUT:
        boundary_verts: list of verts
        cnct_bound_verts: list of list containing the N ordered rings of the mesh
    """
    MIN_NUM_VERTS_RING = 10
    # Ordred dictionary
    edge_dict = get_edges2face(faces)

    boundary_verts = []
    boundary_edges = []
    boundary_faces = []
    for edge, (key, val) in enumerate(edge_dict.items()):
        if len(val.faces) == 1:
            boundary_verts += list(key)
            boundary_edges.append(edge)
            for face_id in val.faces:
                boundary_faces.append(face_id)
    boundary_verts = list(set(boundary_verts))
    if not connected_boundaries:
        return boundary_verts
    n_removed_verts = 0
    if connected_boundaries:
        edge_mat = np.array(list(edge_dict.keys()))
        # Edges on the boundary
        edge_mat = edge_mat[np.array(boundary_edges, dtype=np.int64)]

        # check that every vertex is shared by only two edges
        for v in boundary_verts:
            if np.sum(edge_mat == v) != 2:
                import ipdb; ipdb.set_trace();
                raise ValueError('The boundary edges are not closed loops!')

        cnct_bound_verts = []
        while len(edge_mat > 0):
            # boundary verts, indices of conected boundary verts in order
            bverts = []
            orig_vert = edge_mat[0, 0]
            bverts.append(orig_vert)
            vert = edge_mat[0, 1]
            edge = 0
            while orig_vert != vert:
                bverts.append(vert)
                # remove edge from queue
                edge_mask = np.ones(edge_mat.shape[0], dtype=bool)
                edge_mask[edge] = False
                edge_mat = edge_mat[edge_mask]
                edge = np.where(np.sum(edge_mat == vert, axis=1) > 0)[0]
                tmp = edge_mat[edge]
                vert = tmp[tmp != vert][0]
            # remove the last edge
            edge_mask = np.ones(edge_mat.shape[0], dtype=bool)
            edge_mask[edge] = False
            edge_mat = edge_mat[edge_mask]
            if len(bverts) > MIN_NUM_VERTS_RING:
                # add ring to the list
                cnct_bound_verts.append(bverts)
            else:
                n_removed_verts += len(bverts)
    count = 0
    for ring in cnct_bound_verts: count += len(ring)
    assert(len(boundary_verts) - n_removed_verts == count), "Error computing boundary rings !!"

    if connected_faces:
        return (boundary_verts, boundary_faces, cnct_bound_verts)
    else:
        return (boundary_verts, cnct_bound_verts)


def loop_subdivider(mesh_v, mesh_f):
    """Copied from opendr and modified to work in python3."""

    IS = []
    JS = []
    data = []

    vc = get_vert_connectivity(mesh_v, mesh_f)
    ve = get_vertices_per_edge(mesh_v, mesh_f)
    vo = get_vert_opposites_per_edge(mesh_v, mesh_f)

    if True:
        # New values for each vertex
        for idx in range(len(mesh_v)):

            # find neighboring vertices
            nbrs = np.nonzero(vc[:,idx])[0]

            nn = len(nbrs)

            if nn < 3:
                wt = 0.
            elif nn == 3:
                wt = 3./16.
            elif nn > 3:
                wt = 3. / (8. * nn)
            else:
                raise Exception('nn should be 3 or more')
            if wt > 0.:
                for nbr in nbrs:
                    IS.append(idx)
                    JS.append(nbr)
                    data.append(wt)

            JS.append(idx)
            IS.append(idx)
            data.append(1. - (wt * nn))

    start = len(mesh_v)
    edge_to_midpoint = {}

    if True:
        # New values for each edge:
        # new edge verts depend on the verts they span
        for idx, vs in enumerate(ve):

            vsl = list(vs)
            vsl.sort()
            IS.append(start + idx)
            IS.append(start + idx)
            JS.append(vsl[0])
            JS.append(vsl[1])
            data.append(3./8)
            data.append(3./8)

            opposites = vo[(vsl[0], vsl[1])]
            for opp in opposites:
                IS.append(start + idx)
                JS.append(opp)
                data.append(2./8./len(opposites))

            edge_to_midpoint[(vsl[0], vsl[1])] = start + idx
            edge_to_midpoint[(vsl[1], vsl[0])] = start + idx

    f = []

    for f_i, old_f in enumerate(mesh_f):
        ff = np.concatenate((old_f, old_f))

        for i in range(3):
            v0 = edge_to_midpoint[(ff[i], ff[i+1])]
            v1 = ff[i+1]
            v2 = edge_to_midpoint[(ff[i+1], ff[i+2])]
            f.append(row(np.array([v0,v1,v2])))

        v0 = edge_to_midpoint[(ff[0], ff[1])]
        v1 = edge_to_midpoint[(ff[1], ff[2])]
        v2 = edge_to_midpoint[(ff[2], ff[3])]
        f.append(row(np.array([v0,v1,v2])))

    f = np.vstack(f)

    IS = np.array(IS, dtype=np.uint32)
    JS = np.array(JS, dtype=np.uint32)

    if True: # for x,y,z coords
        IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
        JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
        data = np.concatenate ((data,data,data))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij))

    return mtx, f


def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12.

    Copied from opendr library.
    """

    vpv = sp.csc_matrix((len(mesh_v),len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def get_vertices_per_edge(mesh_v, mesh_f):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()

    Copied from opendr library.
    """

    vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:,0] < result[:,1]] # for uniqueness

    return result


def get_faces_per_edge(mesh_v, mesh_f, verts_per_edge=None):
    """Copied from opendr library."""
    if verts_per_edge is None:
        verts_per_edge = get_vertices_per_edge(mesh_v, mesh_f)

    v2f = {i: set([]) for i in range(len(mesh_v))}
    # TODO: cythonize?
    for idx, f in enumerate(mesh_f):
        v2f[f[0]].add(idx)
        v2f[f[1]].add(idx)
        v2f[f[2]].add(idx)

    fpe = -np.ones_like(verts_per_edge)
    for idx, edge in enumerate(verts_per_edge):
        faces = v2f[edge[0]].intersection(v2f[edge[1]])
        faces = list(faces)[:2]
        for i, f in enumerate(faces):
            fpe[idx,i] = f

    return fpe


def get_vert_opposites_per_edge(mesh_v, mesh_f):
    """Returns a dictionary from vertidx-pairs to opposites.
    For example, a key consist of [4,5)] meaning the edge between
    vertices 4 and 5, and a value might be [10,11] which are the indices
    of the vertices opposing this edge.

    Copied from opendr library.
    """
    result = {}
    for f in mesh_f:
        for i in range(3):
            key = [f[i], f[(i+1)%3]]
            key.sort()
            key = tuple(key)
            val = f[(i+2)%3]

            if key in result:
                result[key].append(val)
            else:
                result[key] = [val]
    return result


if __name__ == '__main__':
    ms = Mesh(filename="/BS/cpatel/work/data/learn_anim/test_py3/t-shirt_male/0000/gt_0.ply")
    b = get_boundary_verts(ms.v, ms.f)
