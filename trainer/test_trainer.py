"""
Implements trainer class for TailorNet low frequency predictor.

LF predictor training is exactly same as MLP baseline training
but one change: smooth_level is set to 1 for LF training.
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
from dataset.static_pose_shape_final import OneStyleShape
import global_var
from . import base_trainer

device = torch.device("cuda:0")
# device = torch.device("cpu")


class TestTrainer(base_trainer.Trainer):
    def load_dataset(self, split):
        params = self.params
        shape_idx, style_idx = params['shape_style'].split('_')

        dataset = OneStyleShape(self.garment_class, shape_idx=shape_idx, style_idx=style_idx, split=split,
                                gender=self.gender, smooth_level=params['smooth_level'])
        shuffle = True if split == 'train' else False
        if split == 'train' and len(dataset) > params['batch_size']:
            drop_last = True
        else:
            drop_last = False
        dataloader = DataLoader(dataset, batch_size=self.bs, num_workers=0, shuffle=shuffle,
                                drop_last=drop_last)
        return dataset, dataloader


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_config', default='')

    parser.add_argument('--garment_class', default="t-shirt")
    parser.add_argument('--gender', default="male")
    parser.add_argument('--shape_style', nargs='+')

    # some training hyper parameters
    parser.add_argument('--vis_freq', default=4096, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    # name under which experiment will be logged
    parser.add_argument('--log_name', default="test_mlp_baseline_singless")

    # smooth_level=1 will train TailorNet low frequency predictor
    parser.add_argument('--smooth_level', default=0, type=int)

    # model specification.
    parser.add_argument('--model_name', default="FullyConnected")
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--hidden_size', default=1024)

    # small experiment description
    parser.add_argument('--note', default="Singe ss baseline")

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
        trainer = TestTrainer(params)

        # try:
        if True:
            for i in range(params['start_epoch'], params['max_epoch']):
                print("epoch: {}".format(i))
                trainer.train(i)
                if (i+1) % 20 == 0:
                    trainer.validate(i)
                # if i % 20 == 0:
                #     trainer.save_ckpt(i)
            # trainer.save_ckpt(params['max_epoch']-1)

            # except Exception as e:
        #     print(str(e))
        # finally:
            trainer.write_log()
            print("safely quit!")


if __name__ == '__main__':
    main()