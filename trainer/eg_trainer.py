"""
Implements trainer class for TailorNet low frequency predictor.

LF predictor training is exactly same as MLP baseline training
but one change: smooth_level is set to 1 for LF training.
"""
import argparse
import json
import os
import torch
import base_trainer
from ss2g_trainer import get_best_runner as ss2g_runner
from models import ops

device = torch.device("cuda:0")
# device = torch.device("cpu")


class EGTrainer(base_trainer.Trainer):
    def __init__(self, params):
        super(EGTrainer, self).__init__(params)
        ss2g_logdir = "/BS/cpatel/work/data/learn_anim/test_ss2g/{}_{}".format(
            self.garment_class, self.gender)
        self.ss2g_runner = ss2g_runner(ss2g_logdir)

    def one_step(self, inputs):
        gt_verts, thetas, betas, gammas, _ = inputs

        thetas = ops.mask_thetas(thetas, self.garment_class)
        gt_verts = gt_verts.to(device)
        thetas = thetas.to(device)
        betas = betas.to(device)
        gammas = gammas.to(device)

        ss2g_verts = self.ss2g_runner.forward(betas=betas, gammas=gammas).view(gt_verts.shape)
        pred_verts = ss2g_verts + self.model(
            torch.cat((thetas, betas, gammas), dim=1)).view(gt_verts.shape)

        # L1 loss
        data_loss = (pred_verts - gt_verts).abs().sum(-1).mean()
        return pred_verts, data_loss


class Runner(base_trainer.Runner):
    def __init__(self, ckpt, params):
        super(Runner, self).__init__(ckpt, params)
        ss2g_logdir = "/BS/cpatel/work/data/learn_anim/test_ss2g/{}_{}".format(
            params['garment_class'], params['gender'])
        self.ss2g_runner = ss2g_runner(ss2g_logdir)

    def forward(self, thetas, betas, gammas):
        pred_verts = super(Runner, self).forward(thetas=thetas, betas=betas, gammas=gammas)
        pred_verts = pred_verts + self.ss2g_runner(betas=betas, gammas=gammas)
        return pred_verts


def get_best_runner(log_dir, epoch_num=None):
    """Returns a trained model runner given the log_dir."""
    ckpt_dir = log_dir
    with open(os.path.join(ckpt_dir, 'params.json')) as jf:
        params = json.load(jf)

    # if epoch_num is not given then pick up the best epoch
    if epoch_num is None:
        with open(os.path.join(ckpt_dir, 'best_epoch')) as f:
            best_epoch = int(f.read().strip())
    else:
        best_epoch = epoch_num
    ckpt_path = os.path.join(ckpt_dir, "{:04d}".format(best_epoch), 'lin.pth.tar')
    ckpt_path = os.path.join(ckpt_dir, 'lin.pth.tar')

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
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    # name under which experiment will be logged
    parser.add_argument('--log_name', default="test_eg_baseline")

    # smooth_level=1 will train TailorNet low frequency predictor
    parser.add_argument('--smooth_level', default=0, type=int)

    # model specification.
    parser.add_argument('--model_name', default="FcModified")
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--hidden_size', default=1024)

    # small experiment description
    parser.add_argument('--note', default="EG baseline")

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
    trainer = EGTrainer(params)

    # try:
    if True:
        for i in range(params['start_epoch'], params['max_epoch']):
            print("epoch: {}".format(i))
            trainer.train(i)
            trainer.validate(i)
            # trainer.save_ckpt(i)

        # except Exception as e:
    #     print(str(e))
    # finally:
        trainer.write_log()
        print("safely quit!")


if __name__ == '__main__':
    main()