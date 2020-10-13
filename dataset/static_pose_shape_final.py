import torch
import os
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import pickle

import global_var
from utils.diffusion_smoothing import DiffusionSmoothing
from models.torch_smpl4garment import TorchSMPL4Garment


# Smoothing levels can be defined here.
# smooth level 0 is not smoothing.
# smooth level 1 is smoothing with 0.15 smoothness for 80 iterations.
level_smoothness = [0, 0.15]
level_smoothiter = [0, 80]
Ltype = "uniform"


def smooth_it(smoothing, smooth_level, smpl, thetas, betas, verts, garment_class):
    """Smoothing function used only when smoothing is done during training time."""
    if smooth_level == -1:
        verts = torch.zeros_like(verts)
    elif smooth_level != 0:
        v_poseshaped = smpl.forward_poseshaped(
            theta=thetas.unsqueeze(0), beta=betas.unsqueeze(0),
            garment_class=garment_class)[0]
        unposed_gar_smooth = (v_poseshaped + verts).numpy()
        unposed_gar_smooth = smoothing.smooth(
            unposed_gar_smooth, smoothness=level_smoothness[smooth_level],
            Ltype=Ltype, n=level_smoothiter[smooth_level])
        verts = torch.from_numpy(unposed_gar_smooth.astype(np.float32)) - v_poseshaped
    return verts


class OneStyleShape(Dataset):
    """Dataset for single pivot style-shape."""
    def __init__(self, garment_class, shape_idx, style_idx, split, gender='female', smooth_level=0):
        super(OneStyleShape, self).__init__()

        self.garment_class = garment_class
        self.split, self.gender = split, gender
        self.style_idx, self.shape_idx = style_idx, shape_idx
        self.smooth_level = smooth_level

        data_dir = os.path.join(global_var.DATA_DIR, '{}_{}'.format(garment_class, gender))

        beta = np.load(os.path.join(data_dir, 'shape/beta_{}.npy'.format(shape_idx)))
        gamma = np.load(os.path.join(data_dir, 'style/gamma_{}.npy'.format(style_idx)))
        # I had a hard time figuring out the bug in the following line:
        # gamma = np.load(os.path.join(data_dir, 'style/gamma_{}.npy'.format(shape_idx)))

        thetas = []
        pose_order = []
        verts_d = []
        smooth_verts_d = []
        seq_idx = 0
        while True:
            seq_path = os.path.join(data_dir, 'pose/{}_{}/poses_{:03d}.npz'.format(shape_idx, style_idx, seq_idx))
            if not os.path.exists(seq_path):
                break
            data = np.load(seq_path)
            verts_d_path = os.path.join(data_dir, 'pose/{}_{}/unposed_{:03d}.npy'.format(shape_idx, style_idx, seq_idx))
            if not os.path.exists(verts_d_path):
                print("{} doesn't exist. This is not an error. "
                      "It's just that this sequence was not simulated well.".format(verts_d_path))
                seq_idx += 1
                continue

            thetas.append(data['thetas'])
            pose_order.append(data['pose_order'])
            verts_d.append(np.load(verts_d_path))

            if smooth_level == 1 and global_var.SMOOTH_STORED:
                smooth_verts_d_path = os.path.join(
                    global_var.SMOOTH_DATA_DIR, '{}_{}'.format(garment_class, gender),
                    'pose/{}_{}/smooth_unposed_{:03d}.npy'.format(shape_idx, style_idx, seq_idx))
                if not os.path.exists(smooth_verts_d_path):
                    print("{} doesn't exist.".format(smooth_verts_d_path))
                    exit(-1)
                smooth_verts_d.append(np.load(smooth_verts_d_path))

            seq_idx += 1
            # print("Using just one sequence file")
            # break

        thetas = np.concatenate(thetas, axis=0)
        pose_order = np.concatenate(pose_order, axis=0)
        verts_d = np.concatenate(verts_d, axis=0)
        if smooth_level == 1 and global_var.SMOOTH_STORED:
            smooth_verts_d = np.concatenate(smooth_verts_d, axis=0)

        if split is not None:
            assert(split in ['test', 'train'])
            # SMPL has 1782 poses. We set aside 350 poses as test set and remaining in train set. So if a frame has a
            # pose from these 1782 poses, it's easy to classify them as train or test.
            # But during simulation of these poses, we add some intermediate poses for simulation stability.
            # To classify these intermediate poses in test and train split, we follow this policy:
            # - For train pivots, intermediate poses go into train set because there are significant amount of
            #   intermediate poses and we can't afford to give them away during training.
            # - For test pivots, we add intermediate poses to test set. Assuming that intermediate poses are randomly
            #   distributed, it's fair to assume that any intermediate test pose will be unseen from training.
            split_file_path = os.path.join(global_var.DATA_DIR, global_var.POSE_SPLIT_FILE)
            if seq_idx > 1:  # train pivot
                test_orig_idx = np.load(split_file_path)['test']
                test_idx = np.in1d(pose_order, test_orig_idx)
                chosen_idx = np.where(test_idx)[0] if split == 'test' else np.where(~test_idx)[0]
            else:  # test pivot
                train_orig_idx = np.load(split_file_path)['train']
                train_idx = np.in1d(pose_order, train_orig_idx)
                chosen_idx = np.where(train_idx)[0] if split == 'train' else np.where(~train_idx)[0]

            thetas = thetas[chosen_idx]
            verts_d = verts_d[chosen_idx]
            if smooth_level == 1 and global_var.SMOOTH_STORED:
                smooth_verts_d = smooth_verts_d[chosen_idx]

        self.verts_d = torch.from_numpy(verts_d.astype(np.float32))
        self.thetas = torch.from_numpy(thetas.astype(np.float32))
        self.beta = torch.from_numpy(beta[:10].astype(np.float32))
        self.gamma = torch.from_numpy(gamma.astype(np.float32))
        if smooth_level == 1 and global_var.SMOOTH_STORED:
            self.smooth_verts_d = torch.from_numpy(smooth_verts_d.astype(np.float32))
            return

        if self.smooth_level != 0 and self.smooth_level != -1:
            with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
                class_info = pickle.load(f)
            num_v = len(class_info[garment_class]['vert_indices'])
            self.smoothing = DiffusionSmoothing(
                np.zeros((num_v, 3)), class_info[garment_class]['f'])
            self.smpl = TorchSMPL4Garment(gender=gender)
        else:
            self.smoothing = None
            self.smpl = None

    def __len__(self):
        return self.thetas.shape[0]

    def __getitem__(self, item):
        verts_d, theta, beta, gamma = self.verts_d[item], self.thetas[item], self.beta, self.gamma
        if self.smooth_level == 1 and global_var.SMOOTH_STORED:
            verts_d = self.smooth_verts_d[item]
        else:
            verts_d = smooth_it(self.smoothing, self.smooth_level, self.smpl,
                                theta, beta, verts_d, self.garment_class)
        return verts_d, theta, beta, gamma, item


class OneStyleShapeHF(OneStyleShape):
    """Dataset for single style-shape which returns GT and smooth garment displacements both.
    It is used to train pivot high frequency predictors.
    """
    def __init__(self, garment_class, shape_idx, style_idx, split, gender='female', smooth_level=0, smpl=None):
        super(OneStyleShapeHF, self).__init__(garment_class, shape_idx, style_idx, split, gender=gender,
                                              smooth_level=smooth_level)
        print("USING HF AS GROUNDTRUTH")

    def __getitem__(self, item):
        verts_d = self.verts_d[item]
        ret = super(OneStyleShapeHF, self).__getitem__(item)
        ret = (verts_d,) + ret
        return ret


class MultiStyleShape(Dataset):
    """Entire dataset for a gender and garment style.
    It creates single style-shape datasets for all pivots and then concate them.
    """
    def __init__(self, garment_class, split=None, gender='female', smooth_level=0, smpl=None):
        super(MultiStyleShape, self).__init__()

        self.garment_class = garment_class
        self.smooth_level = smooth_level
        self.split, self.gender = split, gender
        self.smpl = smpl
        assert(gender in ['neutral', 'male', 'female'])
        assert(split in ['train', 'test', None, 'train_train',
                         'train_test', 'test_train', 'test_test'])

        self.one_style_shape_datasets = self.get_single_datasets()
        self.ds = ConcatDataset(self.one_style_shape_datasets)
        if smooth_level == 1 and global_var.SMOOTH_STORED:
            print("Using Smoothing in the dataset")
            return
        if self.smooth_level != 0 and self.smooth_level != -1:
            print("Using Smoothing in the dataset")
            print(self.smooth_level, Ltype)
            with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
                class_info = pickle.load(f)
            num_v = len(class_info[garment_class]['vert_indices'])
            self.smoothing = DiffusionSmoothing(
                np.zeros((num_v, 3)), class_info[garment_class]['f'])
            self.smpl = TorchSMPL4Garment(gender=gender)
        else:
            self.smoothing = None
            self.smpl = None

    def get_single_datasets(self):
        garment_class, split, gender = self.garment_class, self.split, self.gender
        data_dir = os.path.join(global_var.DATA_DIR, '{}_{}'.format(garment_class, gender))
        with open(os.path.join(data_dir, "pivots.txt"), "r") as f:
            train_pivots = [l.strip().split('_') for l in f.readlines()]

        test_ppath = os.path.join(data_dir, "test.txt")
        if os.path.exists(test_ppath):
            with open(test_ppath, "r") as f:
                test_pivots = [l.strip().split('_') for l in f.readlines()]
        else:
            print("Test pivots not available.")
            test_pivots = []

        single_sl = 0
        if self.smooth_level == 1 and global_var.SMOOTH_STORED:
            single_sl = 1

        one_style_shape_datasets = []
        eval_split = True if split in ['train_train', 'train_test',
                                       'test_train', 'test_test'] else False

        assert test_pivots or not eval_split, "Test split not available yet"

        if not eval_split:
            for shi, sti in train_pivots:
                one_style_shape_datasets.append(
                    OneStyleShape(garment_class, shape_idx=shi, style_idx=sti, split=split, gender=gender,
                                  smooth_level=single_sl))
            for shi, sti in test_pivots:
                if split == 'train': continue
                one_style_shape_datasets.append(
                    OneStyleShape(garment_class, shape_idx=shi, style_idx=sti, split=None, gender=gender,
                                  smooth_level=single_sl))
        else:
            pose_split, shape_split = split.split('_')
            for shi, sti in train_pivots:
                if shape_split == 'test': continue
                one_style_shape_datasets.append(
                    OneStyleShape(garment_class, shape_idx=shi, style_idx=sti, split=pose_split, gender=gender,
                                  smooth_level=single_sl))
            for shi, sti in test_pivots:
                if shape_split == 'train': continue
                one_style_shape_datasets.append(
                    OneStyleShape(garment_class, shape_idx=shi, style_idx=sti, split=pose_split, gender=gender,
                                  smooth_level=single_sl))
        return one_style_shape_datasets

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        verts, thetas, betas, gammas, _ = self.ds[item]
        if self.smooth_level == 1 and global_var.SMOOTH_STORED:
            return verts, thetas, betas, gammas, item
        verts = smooth_it(self.smoothing, self.smooth_level,
                          self.smpl, thetas, betas, verts, self.garment_class)
        return verts, thetas, betas, gammas, item


def visualize():
    from models.smpl4garment import SMPL4Garment

    garment_class = 'shirt'
    gender = 'female'
    split = None
    style_idx = '000'
    shape_idx = '008'
    smooth_level = 1

    smpl = SMPL4Garment(gender=gender)

    # gt_ds = MultiStyleShape(garment_class=garment_class, split=split, smooth_level=smooth_level, gender=gender)
    # gt_ds = OneStyleShape(garment_class=garment_class, shape_idx=shape_idx, style_idx=style_idx, split=split,
    #                       smooth_level=smooth_level, gender=gender)

    gt_ds = MultiStyleShape(garment_class=garment_class, split=None, smooth_level=smooth_level, gender=gender)
    print(len(gt_ds))
    gt_ds = MultiStyleShape(garment_class=garment_class, split='train', smooth_level=smooth_level, gender=gender)
    print(len(gt_ds))
    gt_ds = MultiStyleShape(garment_class=garment_class, split='test', smooth_level=smooth_level, gender=gender)
    print(len(gt_ds))

    for idx in np.random.randint(0, len(gt_ds), 4):

        verts, thetas, betas, gammas, item = gt_ds[idx]
        # verts, sverts, thetas, betas, gammas, item = gt_ds[idx]

        body_m, gar_m = smpl.run(theta=thetas.numpy(), beta=betas.numpy(), garment_class=garment_class,
                                 garment_d=verts.numpy())
        # body_m.write_ply("/BS/cpatel/work/body_{}.ply".format(idx))
        # gar_m.write_ply("/BS/cpatel/work/gar_{}.ply".format(idx))

        # _, gar_m = smpl.run(theta=thetas.numpy(), beta=betas.numpy(), garment_class=garment_class,
        #                     garment_d=sverts.numpy())
        # gar_m.write_ply("/BS/cpatel/work/gars_{}.ply".format(idx))


def save_smooth():
    """Helper function to save smooth garment displacements."""
    garment_class = 'skirt'
    gender = 'female'
    smooth_level = 1
    OUT_DIR = global_var.SMOOTH_DATA_DIR

    data_dir = os.path.join(global_var.DATA_DIR, '{}_{}'.format(garment_class, gender))
    with open(os.path.join(data_dir, "test.txt"), "r") as f:
        train_pivots = [l.strip().split('_') for l in f.readlines()]

    with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
        class_info = pickle.load(f)
    num_v = len(class_info[garment_class]['vert_indices'])
    smoothing = DiffusionSmoothing(
        np.zeros((num_v, 3)), class_info[garment_class]['f'])
    smpl = TorchSMPL4Garment(gender=gender)

    for shape_idx, style_idx in train_pivots:
        beta = torch.from_numpy(np.load(os.path.join(
            data_dir, 'shape/beta_{}.npy'.format(shape_idx))).astype(np.float32)[:10])
        gamma = torch.from_numpy(np.load(os.path.join(
            data_dir, 'style/gamma_{}.npy'.format(shape_idx))).astype(np.float32))
        outdir = os.path.join(OUT_DIR, "{}_{}".format(garment_class, gender), "pose/{}_{}".format(shape_idx, style_idx))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        seq_idx = 0
        while True:
            seq_path = os.path.join(data_dir, 'pose/{}_{}/poses_{:03d}.npz'.format(shape_idx, style_idx, seq_idx))
            if not os.path.exists(seq_path):
                break
            data = np.load(seq_path)
            verts_d_path = os.path.join(data_dir,
                                        'pose/{}_{}/unposed_{:03d}.npy'.format(shape_idx, style_idx, seq_idx))
            if not os.path.exists(verts_d_path):
                print("{} doesn't exist.".format(verts_d_path))
                seq_idx += 1
                continue
            outpath = os.path.join(outdir, "smooth_unposed_{:03d}.npy".format(seq_idx))
            if os.path.exists(outpath):
                print("{} exists.".format(outpath))
                seq_idx += 1
                continue
            print(verts_d_path)
            thetas = torch.from_numpy(data['thetas'].astype(np.float32))
            verts_d = torch.from_numpy(np.load(verts_d_path).astype(np.float32))
            smooth_verts_d = []
            for theta, vert_d in zip(thetas, verts_d):
                svert_d = smooth_it(smoothing, smooth_level, smpl, theta, beta, vert_d, garment_class)
                smooth_verts_d.append(svert_d.numpy())
            smooth_verts_d = np.stack(smooth_verts_d)
            np.save(outpath, smooth_verts_d)

            seq_idx += 1


if __name__ == "__main__":
    # visualize()
    save_smooth()
    pass
