"""
Implements trainer class for TailorNet high frequency predictor.

It overloads base_trainer.Trainer class.
"""
import os
import tensorboardX
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle

from models import networks
from models import ops
from dataset.static_pose_shape_final import OneStyleShapeHF
import global_var
from . import base_trainer

device = torch.device("cuda:0")
# device = torch.device("cpu")


class HFTrainer(base_trainer.Trainer):

    def load_dataset(self, split):
        params = self.params
        shape_idx, style_idx = params['shape_style'].split('_')

        dataset = OneStyleShapeHF(self.garment_class, shape_idx=shape_idx, style_idx=style_idx, split=split,
                                  gender=self.gender, smooth_level=params['smooth_level'])
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
            input_size=72, output_size=self.vert_indices.shape[0] * 3,
            num_layers=params['num_layers'],
            hidden_size=params['hidden_size'])
        return model

    def one_step(self, inputs):
        gt_verts, smooth_verts, thetas, _, _, _ = inputs

        thetas = ops.mask_thetas(thetas, self.garment_class)
        gt_verts = gt_verts.to(device)
        smooth_verts = smooth_verts.to(device)
        thetas = thetas.to(device)
        pred_verts = self.model(thetas).view(gt_verts.shape) + smooth_verts

        # L1 loss
        data_loss = (pred_verts - gt_verts).abs().sum(-1).mean()
        return pred_verts, data_loss

    def update_metrics(self, metrics, inputs, outputs):
        gt_verts = inputs[0]
        pred_verts = outputs
        dist = ops.verts_dist(gt_verts, pred_verts.cpu()) * 1000.
        metrics['val_dist'].update(dist.item(), gt_verts.shape[0])

    def visualize_batch(self, inputs, outputs, epoch):
        # visualize some predictions
        gt_verts, smooth_verts, thetas, betas, gammas, idxs = inputs
        new_inputs = (gt_verts, thetas, betas, gammas, idxs)
        super(HFTrainer, self).visualize_batch(new_inputs, outputs, epoch)


class Runner(object):
    """A helper class to load a trained model."""
    def __init__(self, ckpt, params):
        model_name = params['model_name']
        garment_class = params['garment_class']

        with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
            class_info = pickle.load(f)
        output_size = class_info[garment_class]['vert_indices'].shape[0] * 3

        self.model = getattr(networks, model_name)(
            input_size=72, output_size=output_size,
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

    def forward(self, thetas, betas=None, gammas=None):
        thetas = ops.mask_thetas(thetas=thetas, garment_class=self.garment_class)
        pred_verts = self.model(thetas)
        return pred_verts

    def cuda(self):
        self.model.cuda()

    def to(self, device):
        self.model.to(device)


def get_best_runner(log_dir, epoch_num=None, take_last=False):
    """Returns a trained model runner given the log_dir."""
    ckpt_dir = log_dir
    with open(os.path.join(ckpt_dir, 'params.json')) as jf:
        params = json.load(jf)

    if epoch_num is not None or take_last:
        take_epoch = epoch_num
        if take_last:
            take_epoch = int(params['max_epoch']-1)
        ckpt_path = os.path.join(ckpt_dir, "{:04d}".format(take_epoch), 'lin.pth.tar')
    else:
        ckpt_path = os.path.join(ckpt_dir, 'lin.pth.tar')

    runner = Runner(ckpt_path, params)
    return runner


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_config', default='')

    parser.add_argument('--garment_class', default="t-shirt")
    parser.add_argument('--gender', default="male")
    parser.add_argument('--shape_style', nargs='+')

    # some training hyper parameters
    parser.add_argument('--vis_freq', default=16, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--max_epoch', default=2, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    # name under which experiment will be logged
    parser.add_argument('--log_name', default="test_py3hf")

    # smooth_level=1 will train HF for that smoothness level
    parser.add_argument('--smooth_level', default=1, type=int)

    # model specification.
    parser.add_argument('--model_name', default="FcModified")
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--hidden_size', default=1024)

    # small experiment description
    parser.add_argument('--note', default="TailorNet high frequency prediction")

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
    shape_styles = params['shape_style']

    for ss in shape_styles:
        params['shape_style'] = ss
        print("start training {} on {}".format(params['garment_class'], ss))
        trainer = HFTrainer(params)

        # try:
        if True:
            for i in range(params['start_epoch'], params['max_epoch']):
                print("epoch: {}".format(i))
                trainer.train(i)
                if i % 20 == 0:
                    trainer.validate(i)
                # if i % 20 == 0:
                #     trainer.save_ckpt(i)
            trainer.save_ckpt(params['max_epoch']-1)

            # except Exception as e:
        #     print(str(e))
        # finally:
            trainer.write_log()
            print("safely quit!")


if __name__ == '__main__':
    main()