import os
import sys
import shutil
import json
from datetime import datetime
from os.path import join as opj
import cv2
import struct
import numpy as np
import TailorNet.global_var


def backup_file(src, dst):
    if os.path.exists(opj(src, 'nobackup')):
        return
    if len([k for k in os.listdir(src) if k.endswith('.py') or k.endswith('.sh')]) == 0:
        return
    if not os.path.isdir(dst):
        os.makedirs(dst)
    all_files = os.listdir(src)
    for fname in all_files:
        fname_full = opj(src, fname)
        fname_dst = opj(dst, fname)
        if os.path.isdir(fname_full):
            backup_file(fname_full, fname_dst)
        elif fname.endswith('.py') or fname.endswith('.sh'):
            shutil.copy(fname_full, fname_dst)


def prepare_log_dir(log_name):
    if len(log_name) == 0:
        log_name = datetime.now().strftime("%b%d_%H%M%S")

    log_dir = os.path.join(global_var.LOG_DIR, log_name)
    if not os.path.exists(log_dir):
        print('making %s' % log_dir)
        os.makedirs(log_dir)
    else:
        warning_info = 'Warning: log_dir({}) already exists\n' \
                       'Are you sure continuing? y/n'.format(log_dir)
        if sys.version_info.major == 3:
            a = input(warning_info)
        else:
            a = input(warning_info)
        if a != 'y':
            exit()

    backup_dir = opj(log_dir, 'code')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    backup_file(global_var.ROOT_DIR, backup_dir)
    print("Backup code in {}".format(backup_dir))
    return log_dir


def save_params(log_dir, params, save_name="params"):
    same_num = 1
    while os.path.exists(save_name):
        save_name = save_name + "({})".format(same_num)
    with open(os.path.join(log_dir, save_name+".json"), 'w') as f:
        json.dump(params, f)


def save_pc2(vertices, path):
    # vertices: (N, V, 3), N is the number of frames, V is the number of vertices
    # path: a .pc2 file
    nframes, nverts, _ = vertices.shape
    with open(path, 'wb') as f:
        headerStr = struct.pack('<12siiffi', b'POINTCACHE2\0',
                                1, nverts, 1, 1, nframes)
        f.write(headerStr)
        v = vertices.reshape(-1, 3).astype(np.float32)
        for v_ in v:
            f.write(struct.pack('<fff', v_[0], v_[1], v_[2]))


def read_pc2(path):
    with open(path, 'rb') as f:
        head_fmt = '<12siiffi'
        data_fmt = '<fff'
        head_unpack = struct.Struct(head_fmt).unpack_from
        data_unpack = struct.Struct(data_fmt).unpack_from
        data_size = struct.calcsize(data_fmt)
        headerStr = f.read(struct.calcsize(head_fmt))
        head = head_unpack(headerStr)
        nverts, nframes = head[2], head[5]
        data = []
        for i in range(nverts*nframes):
            data_line = f.read(data_size)
            data.append(list(data_unpack(data_line)))
        data = np.array(data).reshape([nframes, nverts, 3])
    return data
