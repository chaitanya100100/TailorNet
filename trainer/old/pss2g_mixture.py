"""
THIS IS SIMILAR TO ORIGINAL trainer/old/ps2g_mixture_hf.py which was
actually used.
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
from models.smpl4garment import SMPL4Garment
from models.torch_smpl4garment_zhou import TorchSMPL4GarmentZhou
from dataset.static_pose_shape_final import OneStyleShapeHF
from utils.eval import AverageMeter
from utils.logger import PSS2GMixtureLogger
from utils import sio
import global_var

device = torch.device("cuda:0")


class Trainer(object):
    def __init__(self, params):
        self.params = params
        self.gender = params['gender']
        self.garment_class = params['garment_class']

        self.bs = params['batch_size']
        self.vis_freq = params['vis_freq']
        self.model_name = params['model_name']
        self.note = params['note']
        self.style_shape = params['style_shape']
        self.style_idx, self.shape_idx = self.style_shape.split('_')

        # log and backup
        log_name = os.path.join(params['log_name'], self.garment_class)
        self.log_dir = sio.prepare_log_dir(log_name)
        sio.save_params(self.log_dir, params, save_name='params')
        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']

        # smpl for garment
        self.smpl = SMPL4Garment(gender=self.gender)

        # garment specific things
        with open(os.path.join(global_var.DATA_DIR, 'garment_class_info.pkl')) as f:
            class_info = pickle.load(f)
        self.body_f_np = self.smpl.smpl_base.f.astype(np.long)
        self.garment_f_np = class_info[self.garment_class]['f']
        self.garment_f_torch = torch.tensor(self.garment_f_np.astype(np.long)).long().to(device)
        self.vert_indices = class_info[self.garment_class]['vert_indices']

        # dataset and dataloader
        self.train_dataset = OneStyleShapeHF(
            self.garment_class, self.style_idx, self.shape_idx, split='train',
            batch_0_only=False, gender=self.gender, garment_uv_manager=None, smooth_level=1)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.bs, num_workers=12, shuffle=True,
            drop_last=True if len(self.train_dataset) > self.bs else False)

        self.test_dataset = OneStyleShapeHF(
            self.garment_class, self.style_idx, self.shape_idx, split='test',
            batch_0_only=False, gender=self.gender, garment_uv_manager=None, smooth_level=1)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.bs, num_workers=12, shuffle=False,
            drop_last=False)

        print(len(self.train_dataset))
        print(len(self.test_dataset))

        # model and optimizer
        self.model = getattr(networks, self.model_name)(
            input_size=72,
            output_size=self.vert_indices.shape[0] * 3,
            num_layers=params['num_layers'],
            hidden_size=params['hidden_size'])
        self.model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params['lr'], weight_decay=1e-6)

        if params['checkpoint']:
            ckpt_path = params['checkpoint']
            print('loading ckpt from {}'.format(ckpt_path))
            state_dict = torch.load(os.path.join(ckpt_path, 'lin.pth.tar'))
            self.model.load_state_dict(state_dict)
            if os.path.exists(os.path.join(ckpt_path, 'optimizer.pth.tar')):
                state_dict = torch.load(os.path.join(ckpt_path, 'optimizer.pth.tar'))
                self.optimizer.load_state_dict(state_dict)

        self.best_error = np.inf
        self.best_epoch = -1

        self.logger = tensorboardX.SummaryWriter(os.path.join(self.log_dir))
        self.csv_logger = PSS2GMixtureLogger()

    def get_loss(self, pred_verts, gt_verts, thetas, betas, gammas, tttype, epoch):
        data_loss = (pred_verts - gt_verts).abs().sum(-1).mean()
        return data_loss

    def train(self, epoch):
        epoch_loss = AverageMeter()
        self.model.train()
        for i, (gt_verts, smooth_verts,
                thetas, betas, gammas, _) in enumerate(self.train_loader):
            thetas = ops.mask_thetas(thetas, self.garment_class)
            gt_verts = gt_verts.to(device)
            smooth_verts = smooth_verts.to(device)
            thetas = thetas.to(device)
            betas = betas.to(device)
            gammas = gammas.to(device)

            self.optimizer.zero_grad()
            pred_verts = self.model(thetas).view(gt_verts.shape) + smooth_verts

            loss = self.get_loss(pred_verts, gt_verts,
                                 thetas, betas, gammas, "train", epoch)
            loss.backward()
            self.optimizer.step()

            self.logger.add_scalar("train/loss", loss.item(), self.iter_nums)
            print("Iter {}, loss: {:.8f}".format(self.iter_nums, loss.item()))
            epoch_loss.update(loss, gt_verts.shape[0])
            self.iter_nums += 1

        self.logger.add_scalar("train_epoch/loss", epoch_loss.avg, epoch)
        # self._save_ckpt(epoch)

    def validate(self, epoch):
        val_loss = AverageMeter()
        val_dist = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (gt_verts, smooth_verts, thetas,
                    betas, gammas, idxs) in enumerate(self.test_loader):
                thetas = ops.mask_thetas(thetas, self.garment_class)
                gt_verts = gt_verts.to(device)
                smooth_verts = smooth_verts.to(device)
                thetas = thetas.to(device)
                betas = betas.to(device)
                gammas = gammas.to(device)

                idxs = idxs.numpy()

                pred_verts = self.model(thetas).view(gt_verts.shape) + smooth_verts

                loss = self.get_loss(
                    pred_verts, gt_verts, thetas, betas, gammas, "val", epoch)
                dist = ops.verts_dist(gt_verts, pred_verts) * 1000.

                val_loss.update(loss.item(), gt_verts.shape[0])
                val_dist.update(dist.item(), gt_verts.shape[0])

                for lidx, idx in enumerate(idxs):
                    if idx % self.vis_freq != 0:
                        continue
                    theta = thetas[lidx].cpu().numpy()
                    beta = betas[lidx].cpu().numpy()
                    pred_vert = pred_verts[lidx].cpu().numpy()
                    gt_vert = gt_verts[lidx].cpu().numpy()
                    # linear_vert = linear_pred[lidx].cpu().numpy()

                    body_m, pred_m = self.smpl.run(theta=theta, garment_d=pred_vert, beta=beta,
                                                   garment_class=self.garment_class)
                    _, gt_m = self.smpl.run(theta=theta, garment_d=gt_vert, beta=beta,
                                            garment_class=self.garment_class)
                    # _, linear_m = self.smpl.run(theta=theta, garment_d=linear_vert, beta=beta,
                    #                             garment_class=self.garment_class)

                    save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
                    pred_m.write_ply(os.path.join(save_dir, "pred_{}.ply".format(idx)))
                    gt_m.write_ply(os.path.join(save_dir, "gt_{}.ply".format(idx)))
                    # linear_m.write_ply(os.path.join(save_dir, "linear_pred_{}.ply".format(idx)))
                    body_m.write_ply(os.path.join(save_dir, "body_{}.ply".format(idx)))

        self.logger.add_scalar("val/loss", val_loss.avg, epoch)
        self.logger.add_scalar("val/dist", val_dist.avg, epoch)
        # if epoch > 500:
        #     self._save_ckpt(epoch)

        print("VALIDATION")
        print("Epoch {}, loss: {:.4f}, dist: {:.4f} mm".format(
            epoch, val_loss.avg, val_dist.avg))

        if val_dist.avg < self.best_error:
            self.best_error = val_dist.avg
            self.best_epoch = epoch
            self._save_ckpt_best()
            with open(os.path.join(self.log_dir, 'best_epoch'), 'w') as f:
                f.write("{:04d}".format(epoch))

    def write_log(self):
        if self.best_epoch >= 0:
            self.csv_logger.add_item(
                best_error=self.best_error, best_epoch=self.best_epoch, **self.params)

    def _save_ckpt(self, epoch):
        save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(),
                   os.path.join(save_dir, 'lin.pth.tar'))
        torch.save(self.optimizer.state_dict(),
                   os.path.join(save_dir, "optimizer.pth.tar"))

    def _save_ckpt_best(self):
        save_dir = self.log_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(),
                   os.path.join(save_dir, 'lin.pth.tar'))
        torch.save(self.optimizer.state_dict(),
                   os.path.join(save_dir, "optimizer.pth.tar"))


class RunnerSingle(object):
    def __init__(self, ckpt, params):
        model_name = params['model_name']
        garment_class = params['garment_class']

        with open(os.path.join(global_var.DATA_DIR, 'garment_class_info.pkl')) as f:
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


def get_best_runner_single(log_dir, epoch_num=None):
    ckpt_dir = log_dir
    with open(os.path.join(ckpt_dir, 'params.json')) as jf:
        params = json.load(jf)

    if epoch_num is None:
        with open(os.path.join(ckpt_dir, 'best_epoch')) as f:
            best_epoch = int(f.read().strip())
    else:
        best_epoch = epoch_num
    ckpt_path = os.path.join(ckpt_dir, "{:04d}".format(best_epoch), 'lin.pth.tar')

    runner = RunnerSingle(ckpt_path, params)
    return runner


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_config', default='')

    parser.add_argument('--garment_class', default="smooth_TShirtNoCoat")
    parser.add_argument('--gender', default="female")

    parser.add_argument('--vis_freq', default=16, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--max_epoch', default=800, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    parser.add_argument('--model_name', default="FcModified")
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--hidden_size', default=1024)
    parser.add_argument('--note', default="predict hf, uniform(0.15, 80), FcModified, 3 layers, 1e-6 decay")

    parser.add_argument('--style_shape', nargs='+')
    parser.add_argument('--log_name', default="mixture_exp34")
    args = parser.parse_args()

    params = args.__dict__

    if os.path.exists(params['local_config']):
        print("loading config from {}".format(params['local_config']))
        with open(params['local_config']) as f:
            lc = json.load(f)
        for k, v in lc.items():
            params[k] = v
    return params


if __name__ == '__main__':
    params = parse_argument()

    style_shapes = params['style_shape']
    log_name = params['log_name']
    garment_class = params['garment_class']

    for ss in style_shapes:
        params['style_shape'] = ss
        print(ss)

        print("start training {}".format(garment_class))
        start_epoch = params['start_epoch']

        params['log_name'] = os.path.join(log_name, params['style_shape'])
        trainer = Trainer(params)
        try:
        # if True:
            for i in range(start_epoch, params['max_epoch']):
                print("epoch: {}".format(i))
                trainer.train(i)
                if i % 10 == 0:
                    trainer.validate(i)
        finally:
        # else:
            trainer.write_log()
            print("safely quit!")