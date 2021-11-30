# -*- coding: UTF-8 -*-
import numpy as np
import os
import copy

BLEND_FILE = "./visualization/scene.blend"

def blender_render(meshes_path, tex_num, outpath):
    import bpy

    scene = bpy.context.scene
    # scene.render.engine = 'CYCLES'

    bpy.context.scene.render.resolution_x = 2048
    bpy.context.scene.render.resolution_y = 2048

    scene.render.resolution_percentage = 100
    scene.render.use_border = False

    scene.render.alpha_mode = 'TRANSPARENT'
    names = []
    for idx in range(len(meshes_path)):
        # imported_object = bpy.ops.import_mesh.ply(filepath=meshes_path[idx])
        imported_object = bpy.ops.import_scene.obj(filepath=meshes_path[idx], axis_forward='-Z', axis_up='Y')
        obj_object = bpy.context.selected_objects[0]  ####<--Fix
        names.append(obj_object.name)

        # smooth shading
        for f in obj_object.data.polygons:
            f.use_smooth = True

        print(tex_num[idx])
        print("NO SHADOWS")
        if tex_num[idx] == 0: # body
            mat = bpy.data.materials['Material.007']
            # mat.use_cast_shadows = False
        elif tex_num[idx] == 1: # tshirt
            mat = bpy.data.materials['Material.006']
            # mat.use_cast_shadows = False
        elif tex_num[idx] == 2:  # shirt
            mat = bpy.data.materials['Material.006']
            mat = mat.copy()
            mat.diffuse_color = (0.6, 1, 0.8)
        elif tex_num[idx] == 3:  # Pants
            mat = bpy.data.materials['Material.006']
            mat = mat.copy()
            mat.diffuse_color = (1.0, 0.4, 0.4)
        elif tex_num[idx] == 4:  # skirt
            mat = bpy.data.materials['Material.006']
            mat = mat.copy()
            mat.diffuse_color = (0.35, 0.4, 1.0)
        else:
            raise AttributeError
        obj_object.data.materials.append(mat)
        bpy.ops.object.shade_smooth()

    bpy.context.scene.render.filepath = outpath
    bpy.ops.render.render(write_still=True)
    for nm in names:
        objs = bpy.data.objects
        objs.remove(objs[nm], do_unlink=True)


def get_rotmat(side):
    from scipy.spatial.transform import Rotation as R
    if side == 'front':
        s = R.from_rotvec((0.) * np.array([1,0,0]))
    elif side == 'back':
        s = R.from_rotvec((np.pi) * np.array([0,1,0]))
    elif side.startswith("right"):
        angle = side.replace("right", "")
        s = R.from_rotvec((float(angle) * np.pi / 180) * np.array([0,1,0]))
    return s.as_dcm()

    
def preproc_garbody(gar, body, garment_class=None, side='front'):
    gar = copy.copy(gar)
    body = copy.copy(body)

    rotmat = get_rotmat(side)
    gar.v = gar.v.dot(rotmat)
    body.v = body.v.dot(rotmat)

    miny = body.v[:, 1].min() + 0.72

    gar.v[:, 1] -= miny
    body.v[:, 1] -= miny
    return gar, body


def visualize_garment_body(gar, body, outpath, garment_class='t-shirt', side='front'):
    gar = copy.copy(gar)
    body = copy.copy(body)

    rotmat = get_rotmat(side)
    gar.v = gar.v.dot(rotmat)
    body.v = body.v.dot(rotmat)

    miny = body.v[:, 1].min() + 0.72

    gar.v[:, 1] -= miny
    body.v[:, 1] -= miny

    gar_path = "/tmp/gar.obj"
    body_path = "/tmp/body.obj"
    if gar_path.endswith("obj"):
        gar.write_obj(gar_path)
        body.write_obj(body_path)
    else:
        gar.write_ply(gar_path)
        body.write_ply(body_path)

    thispath = os.path.abspath(__file__)
    cmd = "blender --background {} -P {} -- --body {} --gar {} --outpath {} --gar_classes {}".format(
            BLEND_FILE,
            thispath,
            body_path,
            gar_path,
            outpath,
            garment_class,
            )
    os.system(cmd)
    print("Done")


def visualize_two_garments_body(lower, upper, body, outpath, lower_gc=None, upper_gc=None, side='front'):
    lower = copy.copy(lower)
    upper = copy.copy(upper)
    body = copy.copy(body)

    rotmat = get_rotmat(side)
    lower.v = lower.v.dot(rotmat)
    upper.v = upper.v.dot(rotmat)
    body.v = body.v.dot(rotmat)

    miny = body.v[:, 1].min() + 0.72

    lower.v[:, 1] -= miny
    upper.v[:, 1] -= miny
    body.v[:, 1] -= miny

    lower_path = "/tmp/lower.obj"
    upper_path = "/tmp/upper.obj"
    body_path = "/tmp/body.obj"
    if body_path.endswith("obj"):
        lower.write_obj(lower_path)
        upper.write_obj(upper_path)
        body.write_obj(body_path)
    else:
        lower.write_ply(lower_path)
        upper.write_ply(upper_path)
        body.write_ply(body_path)

    thispath = os.path.abspath(__file__)
    cmd = "blender --background {} -P {}  -- --body {} --gar {} {} --outpath {} --gar_classes {} {}".format(
            BLEND_FILE,
            thispath,
            body_path,
            lower_path,
            upper_path,
            outpath,
            lower_gc,
            upper_gc,
            )
    os.system(cmd)
    print("Done")


def visualize_garment(gar, outpath, garment_class='t-shirt', side='front'):
    import os
    import copy
    gar = copy.copy(gar)

    rotmat = get_rotmat(side)
    gar.v = gar.v.dot(rotmat)

    gar_path = "/tmp/gar.obj"
    if gar_path.endswith("obj"):
        gar.write_obj(gar_path)
    else:
        gar.write_ply(gar_path)

    thispath = os.path.abspath(__file__)
    cmd = "blender --background {} -P {} -- --gar {} --outpath {} --gar_classes {}".format(
        BLEND_FILE,
        thispath,
        gar_path,
        outpath,
        garment_class,
    )
    os.system(cmd)
    print("Done")


def visualize_body(body, outpath, side='front'):
    import os
    import copy
    body = copy.copy(body)

    rotmat = get_rotmat(side)
    body.v = body.v.dot(rotmat)

    thispath = os.path.abspath(__file__)
    body_path = "/tmp/body.obj"
    if body_path.endswith("obj"):
        body.write_obj(body_path)
    else:
        body.write_ply(body_path)

    cmd = "blender --background {} -P {} -- --body {} --outpath {}".format(
        BLEND_FILE,
        thispath,
        body_path,
        outpath,
        side,
    )
    os.system(cmd)
    print("Done")


if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0]] + sys.argv[6:]
    print(sys.argv)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath')
    parser.add_argument('--body', nargs='+')
    parser.add_argument('--gar', nargs='+')
    parser.add_argument('--gar_classes', nargs='+')
    args = parser.parse_args()
    print(args)

    # assert(len(args.body) == len(args.gar))
    tex_num = []
    meshes = []

    dd = {
        't-shirt': 1,
        'old-t-shirt': 1,
        'shirt': 2,
        'pant': 3,
        'skirt': 4,
        'short-pant': 3,
    }

    if args.body is not None:
        tex_num += [0]*len(args.body)
        meshes += args.body
    if args.gar is not None:
        meshes += args.gar
        if args.gar_classes is None:
            tex_num += [1] * len(args.gar)
        else:
            tex_num += [dd[gc] for gc in args.gar_classes]
    blender_render(meshes, tex_num=tex_num, outpath=args.outpath)
