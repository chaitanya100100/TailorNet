import numpy as np
import os
import global_var


def get_style(gender, garment_class, style_idx):
    fpath = os.path.join(global_var.DATA_DIR, "{}_{}".format(garment_class, gender),
                         "style/gamma_{}.npy".format(style_idx))
    return np.load(fpath)


def get_shape(gender, garment_class, shape_idx):
    fpath = os.path.join(global_var.DATA_DIR, "{}_{}".format(garment_class, gender),
                         "shape/beta_{}.npy".format(shape_idx))
    return np.load(fpath)


def get_specific_pose(which):
    garment_class = 'smooth_TShirtNoCoat'
    gender = 'female'
    if garment_class == 'smooth_TShirtNoCoat':
        thetas = np.load(os.path.join("/BS/cloth-anim/work/data/md/",
                                      '{}_{}_pcagen_staticshape'.format(gender, garment_class),
                                      '000', 'poses.npz'))['thetas']
        if which == 'apose':
           from utils.rotation import get_Apose
           theta = get_Apose().astype(np.float32)
        elif which == '0':
           theta = thetas[44]
        elif which == '1':
           theta = thetas[75]
        elif which == '2':
           theta = thetas[118]
        elif which == '3':
           theta = thetas[122]
        elif which == '4':
           theta = thetas[196]
        elif which == '5':
           theta = thetas[449]
        elif which == '6':
           theta = thetas[480]
        elif which == '7':
           theta = thetas[767]
        elif which == '8':
           theta = thetas[867]
        elif which == '9':
           theta = thetas[1255]
        elif which == '10':
           theta = thetas[2329]
        else:
           raise AttributeError
    else:
        raise AttributeError
    return theta
