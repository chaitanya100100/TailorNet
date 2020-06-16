"""
Implements trainer class for TailorNet low frequency predictor.

LF predictor training is exactly same as MLP baseline training
but one change: smooth_level is set to 1 for LF training.
"""
import argparse
import json
import os
import torch
from trainer import base_trainer

device = torch.device("cuda:0")
# device = torch.device("cpu")


class LFTrainer(base_trainer.Trainer):
    pass


def get_best_runner(log_dir, epoch_num=None):
    return base_trainer.get_best_runner(log_dir, epoch_num)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_config', default='')

    parser.add_argument('--garment_class', default="t-shirt")
    parser.add_argument('--gender', default="male")
    parser.add_argument('--shape_style', default="")

    # some training hyper parameters
    parser.add_argument('--vis_freq', default=512, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    # name under which experiment will be logged
    parser.add_argument('--log_name', default="test_py3_lf")

    # smooth_level=1 will train TailorNet low frequency predictor
    parser.add_argument('--smooth_level', default=1, type=int)

    # model specification.
    parser.add_argument('--model_name', default="FullyConnected")
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--hidden_size', default=1024)

    # small experiment description
    parser.add_argument('--note', default="TailorNet low frequency prediction")

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

    print("start training {}".format(params['garment_class']))
    trainer = LFTrainer(params)

    for i in range(params['start_epoch'], params['max_epoch']):
        print("epoch: {}".format(i))
        trainer.train(i)
        trainer.validate(i)
        # trainer.save_ckpt(i)

    trainer.write_log()
    print("safely quit!")


if __name__ == '__main__':
    main()
