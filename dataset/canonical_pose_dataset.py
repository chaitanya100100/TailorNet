import os
import numpy as np
import torch
from torch.utils.data import Dataset

import global_var
from utils.rotation import get_Apose


class ShapeStyleCanonPose(Dataset):
    def __init__(self, garment_class, gender, shape_style_list_path='avail.txt', split=None):
        super(ShapeStyleCanonPose, self).__init__()
        self.garment_class = garment_class
        self.gender = gender
        root_dir = os.path.join(global_var.DATA_DIR, '{}_{}'.format(garment_class, gender))

        betas = np.load(os.path.join(root_dir, 'shape/betas.npy'))[:, :10]
        gammas = np.load(os.path.join(root_dir, 'style/gammas.npy'))

        with open(os.path.join(root_dir, shape_style_list_path), "r") as f:
            ss_list = [l.strip().split('_') for l in f.readlines()]

        assert(split in [None, 'train', 'test'])
        if split == 'train':
            ss_list = ss_list[:-20]
        elif split == 'test':
            ss_list = ss_list[-20:]

        unpose_v = []
        for shape_idx, style_idx in ss_list:
            fpath = os.path.join(
                root_dir, 'style_shape/beta{}_gamma{}.npy'.format(shape_idx, style_idx))
            if not os.path.exists(fpath):
                print("shape {} and style {} not available".format(shape_idx, style_idx))
            unpose_v.append(np.load(fpath))
        unpose_v = np.stack(unpose_v)

        self.ss_list = ss_list
        self.betas = torch.from_numpy(betas.astype(np.float32))
        self.gammas = torch.from_numpy(gammas.astype(np.float32))
        self.unpose_v = torch.from_numpy(unpose_v.astype(np.float32))
        self.apose = torch.from_numpy(get_Apose().astype(np.float32))

    def __len__(self):
        return self.unpose_v.shape[0]

    def __getitem__(self, item):
        bi, gi = self.ss_list[item]
        bi, gi = int(bi), int(gi)
        return self.unpose_v[item], self.apose, self.betas[bi], self.gammas[gi], item


if __name__ == '__main__':
    gender = 'male'
    garment_class = 't-shirt'
    ds = ShapeStyleCanonPose(gender=gender, garment_class=garment_class)
    for i in range(len(ds)):
        bla = ds[i]
        print(i)
