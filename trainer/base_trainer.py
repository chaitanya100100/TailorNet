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
from dataset.static_pose_shape_final import MultiStyleShape
from utils.eval import AverageMeter
from utils.logger import TailorNetLogger
from utils import sio
import global_var

device = torch.device("cuda:0")
# device = torch.device("cpu")


class Trainer(object):
    """Implements trainer class for TailorNet MLP baseline.
    It is also a base class for TailorNet LF, HF and SS2G trainers.
    """

    def __init__(self, params):
        self.params = params
        self.gender = params['gender']
        self.garment_class = params['garment_class']

        self.bs = params['batch_size']
        self.vis_freq = params['vis_freq']
        self.model_name = params['model_name']
        self.note = params['note']

        # log and backup
        log_name = os.path.join(params['log_name'],
                                '{}_{}'.format(self.garment_class, self.gender))
        if params['shape_style'] != '':
            log_name = os.path.join(log_name, params['shape_style'])
        self.log_dir = sio.prepare_log_dir(log_name)
        sio.save_params(self.log_dir, params, save_name='params')

        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']

        # smpl for garment
        self.smpl = SMPL4Garment(gender=self.gender)

        # garment specific things
        with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
            class_info = pickle.load(f)
        self.body_f_np = self.smpl.smpl_base.f.astype(np.long)
        self.garment_f_np = class_info[self.garment_class]['f']
        self.garment_f_torch = torch.tensor(self.garment_f_np.astype(np.long)).long().to(device)
        self.vert_indices = class_info[self.garment_class]['vert_indices']

        # get dataset and dataloader
        self.train_dataset, self.train_loader = self.load_dataset('train')
        self.test_dataset, self.test_loader = self.load_dataset('test')
        print("Train dataset size", len(self.train_dataset))
        print("Test dataset size", len(self.test_dataset))

        # model and optimizer
        self.model = self.build_model()
        self.model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        # continue training from checkpoint if provided
        if params['checkpoint']:
            ckpt_path = params['checkpoint']
            print('loading ckpt from {}'.format(ckpt_path))
            state_dict = torch.load(os.path.join(ckpt_path, 'lin.pth.tar'))
            self.model.load_state_dict(state_dict)
            state_dict = torch.load(os.path.join(ckpt_path, 'optimizer.pth.tar'))
            self.optimizer.load_state_dict(state_dict)

        self.best_error = np.inf
        self.best_epoch = -1

        # logger
        self.logger = tensorboardX.SummaryWriter(os.path.join(self.log_dir))
        self.csv_logger = self.get_logger()

    def load_dataset(self, split):
        params = self.params
        dataset = MultiStyleShape(self.garment_class, split=split, gender=self.gender,
                                  smooth_level=params['smooth_level'])
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
            input_size=72+10+4, output_size=self.vert_indices.shape[0] * 3,
            num_layers=params['num_layers'],
            hidden_size=params['hidden_size'])
        return model

    def get_logger(self):
        return TailorNetLogger()

    def one_step(self, inputs):
        """One forward pass.
        Takes `inputs` tuple. Returns output(s) and loss.
        """
        gt_verts, thetas, betas, gammas, _ = inputs

        thetas, betas, gammas = ops.mask_inputs(thetas, betas, gammas, self.garment_class)
        gt_verts = gt_verts.to(device)
        thetas = thetas.to(device)
        betas = betas.to(device)
        gammas = gammas.to(device)
        pred_verts = self.model(
            torch.cat((thetas, betas, gammas), dim=1)).view(gt_verts.shape)

        # L1 loss
        data_loss = (pred_verts - gt_verts).abs().sum(-1).mean()
        return pred_verts, data_loss

    def train(self, epoch):
        """Train for one epoch."""
        epoch_loss = AverageMeter()
        self.model.train()
        for i, inputs in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            outputs, loss = self.one_step(inputs)
            loss.backward()
            self.optimizer.step()

            self.logger.add_scalar("train/loss", loss.item(), self.iter_nums)
            print("Iter {}, loss: {:.8f}".format(self.iter_nums, loss.item()))
            epoch_loss.update(loss, inputs[0].shape[0])
            self.iter_nums += 1

        self.logger.add_scalar("train_epoch/loss", epoch_loss.avg, epoch)

    def update_metrics(self, metrics, inputs, outputs):
        """Update metrics from inputs and predicted outputs."""
        gt_verts = inputs[0]
        pred_verts = outputs
        dist = ops.verts_dist(gt_verts, pred_verts.cpu()) * 1000.
        metrics['val_dist'].update(dist.item(), gt_verts.shape[0])

    def visualize_batch(self, inputs, outputs, epoch):
        """Save visualizations of some samples of the batch."""
        gt_verts, thetas, betas, gammas, idxs = inputs
        pred_verts = outputs
        idxs = idxs.numpy()
        for lidx, idx in enumerate(idxs):
            if idx % self.vis_freq != 0:
                continue
            theta = thetas[lidx].cpu().numpy()
            beta = betas[lidx].cpu().numpy()
            pred_vert = pred_verts[lidx].cpu().numpy()
            gt_vert = gt_verts[lidx].cpu().numpy()

            body_m, pred_m = self.smpl.run(theta=theta, garment_d=pred_vert, beta=beta,
                                           garment_class=self.garment_class)
            _, gt_m = self.smpl.run(theta=theta, garment_d=gt_vert, beta=beta,
                                    garment_class=self.garment_class)

            save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
            pred_m.write_ply(os.path.join(save_dir, "pred_{}.ply".format(idx)))
            gt_m.write_ply(os.path.join(save_dir, "gt_{}.ply".format(idx)))
            body_m.write_ply(os.path.join(save_dir, "body_{}.ply".format(idx)))

    def validate(self, epoch):
        """Evaluate on test dataset."""
        val_loss = AverageMeter()
        metrics = {
            'val_dist': AverageMeter(), # per vertex distance in mm
        }
        self.model.eval()
        with torch.no_grad():
            for i, inputs in enumerate(self.test_loader):
                outputs, loss = self.one_step(inputs)
                val_loss.update(loss.item(), inputs[0].shape[0])

                self.update_metrics(metrics, inputs, outputs)
                self.visualize_batch(inputs, outputs, epoch)

        val_dist_avg = metrics['val_dist'].avg
        self.logger.add_scalar("val/loss", val_loss.avg, epoch)
        self.logger.add_scalar("val/dist", val_dist_avg, epoch)
        print("VALIDATION")
        print("Epoch {}, loss: {:.4f}, dist: {:.4f} mm".format(
            epoch, val_loss.avg, val_dist_avg))

        if val_dist_avg < self.best_error:
            self.best_error = val_dist_avg
            self.best_epoch = epoch
            self.save_ckpt_best()
            with open(os.path.join(self.log_dir, 'best_epoch'), 'w') as f:
                f.write("{:04d}".format(epoch))

    def write_log(self):
        """Log training info once training is done."""
        if self.best_epoch >= 0:
            self.csv_logger.add_item(
                best_error=self.best_error, best_epoch=self.best_epoch, **self.params)

    def save_ckpt(self, epoch):
        """Save checkpoint in given epoch's directory."""
        save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'lin.pth.tar'))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth.tar"))

    def save_ckpt_best(self):
        """Save checkpoint in log directory."""
        save_dir = self.log_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'lin.pth.tar'))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth.tar"))


class Runner(object):
    """A helper class to load a trained model."""
    def __init__(self, ckpt, params):
        model_name = params['model_name']
        garment_class = params['garment_class']

        with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
            class_info = pickle.load(f)
        output_size = class_info[garment_class]['vert_indices'].shape[0] * 3

        self.model = getattr(networks, model_name)(
            input_size=72+10+4, output_size=output_size,
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

    def forward(self, thetas, betas, gammas):
        thetas, betas, gammas = ops.mask_inputs(
            thetas, betas, gammas, garment_class=self.garment_class)
        pred_verts = self.model(torch.cat((thetas, betas, gammas), dim=1))
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
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    # name under which experiment will be logged
    parser.add_argument('--log_name', default="tn_baseline")

    # smooth_level=0 will train TailorNet MLP baseline
    parser.add_argument('--smooth_level', default=0, type=int)

    # model specification.
    parser.add_argument('--model_name', default="FullyConnected")
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--hidden_size', default=1048)

    # small experiment description
    parser.add_argument('--note', default="MLP Baseline")

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
    trainer = Trainer(params)

    for i in range(params['start_epoch'], params['max_epoch']):
        print("epoch: {}".format(i))
        trainer.train(i)
        trainer.validate(i)
        # trainer.save_ckpt(i)

    trainer.write_log()
    print("safely quit!")


if __name__ == '__main__':
    main()
