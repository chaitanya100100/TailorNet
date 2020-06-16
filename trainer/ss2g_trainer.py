"""
Implements trainer class to predict deformations in canonical pose.

It overloads base_trainer.Trainer class. This predictor is used in
TailorNet to get the weights of pivot high frequency outputs.
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
import json
import pickle

from models import networks
from dataset.canonical_pose_dataset import ShapeStyleCanonPose
import global_var
from . import base_trainer

device = torch.device("cuda:0")
# device = torch.device("cpu")


class SS2GTrainer(base_trainer.Trainer):

    def load_dataset(self, split):
        params = self.params
        dataset = ShapeStyleCanonPose(self.garment_class, split=split, gender=self.gender)
        shuffle = True if split == 'train' else False
        if split == 'train' and len(dataset) > params['batch_size']:
            drop_last = True
        else:
            drop_last = False
        dataloader = DataLoader(dataset, batch_size=self.bs, num_workers=0, shuffle=shuffle,
                                drop_last=drop_last)
        return dataset, dataloader

    def build_model(self):
        params = self.params
        model = getattr(networks, self.model_name)(
            input_size=10+4, output_size=self.vert_indices.shape[0] * 3,
            num_layers=params['num_layers'],
            hidden_size=params['hidden_size'])
        return model

    def one_step(self, inputs):
        gt_verts, _, betas, gammas, _ = inputs

        gt_verts = gt_verts.to(device)
        betas = betas.to(device)
        gammas = gammas.to(device)
        pred_verts = self.model(
            torch.cat((betas, gammas), dim=1)).view(gt_verts.shape)

        # L1 loss
        data_loss = (pred_verts - gt_verts).abs().sum(-1).mean()
        return pred_verts, data_loss


class Runner(object):
    """A helper class to load a trained model."""
    def __init__(self, ckpt, params):
        model_name = params['model_name']
        garment_class = params['garment_class']

        with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
            class_info = pickle.load(f)
        output_size = class_info[garment_class]['vert_indices'].shape[0] * 3

        self.model = getattr(networks, model_name)(
            input_size=10+4, output_size=output_size,
            hidden_size=params['hidden_size'] if 'hidden_size' in params else 1024,
            num_layers=params['num_layers'] if 'num_layers' in params else 3
        )
        self.garment_class = params['garment_class']

        print("loading {}".format(ckpt))
        if torch.cuda.is_available():
            self.model.cuda()
            state_dict = torch.load(ckpt)
        else:
            state_dict = torch.load(ckpt,map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, thetas=None, betas=None, gammas=None):
        pred_verts = self.model(torch.cat((betas, gammas), dim=1))
        return pred_verts

    def cuda(self):
        self.model.cuda()

    def to(self, device):
        self.model.to(device)


def get_best_runner(log_dir, epoch_num=None):
    """Returns a trained model runner given the log_dir."""
    ckpt_dir = log_dir
    with open(os.path.join(ckpt_dir, 'params.json')) as jf:
        params = json.load(jf)

    # if epoch_num is not given then pick up the best epoch
    if epoch_num is None:
        ckpt_path = os.path.join(ckpt_dir, 'lin.pth.tar')
    else:
        # with open(os.path.join(ckpt_dir, 'best_epoch')) as f:
        #     best_epoch = int(f.read().strip())
        best_epoch = epoch_num
        ckpt_path = os.path.join(ckpt_dir, "{:04d}".format(best_epoch), 'lin.pth.tar')

    runner = Runner(ckpt_path, params)
    return runner


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_config', default='')

    parser.add_argument('--garment_class', default="t-shirt")
    parser.add_argument('--gender', default="male")
    parser.add_argument('--shape_style', default="")

    # some training hyper parameters
    parser.add_argument('--vis_freq', default=512, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    # name under which experiment will be logged
    parser.add_argument('--log_name', default="test_py3ss2g")

    # smooth_level=0 will train TailorNet MLP baseline
    parser.add_argument('--smooth_level', default=0, type=int)

    # model specification.
    parser.add_argument('--model_name', default="FullyConnected")
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--hidden_size', default=128)

    # small experiment description
    parser.add_argument('--note', default="SS2G training")

    args = parser.parse_args()
    params = args.__dict__

    # load params from local config if provided
    if os.path.exists(params['local_config']):
        print("loading config from {}".format(params['local_config']))
        with open(params['local_config']) as f:
            lc = json.load(f)
        for k, v in lc.items():
            params[k] = v
    return params


def main():
    params = parse_argument()

    print("start training ss2g {}".format(params['garment_class']))
    trainer = SS2GTrainer(params)

    # try:
    if True:
        for i in range(params['start_epoch'], params['max_epoch']):
            print("epoch: {}".format(i))
            trainer.train(i)
            trainer.validate(i)

        # except Exception as e:
    #     print(str(e))
    # finally:
        trainer.write_log()
        print("safely quit!")


if __name__ == '__main__':
    main()
