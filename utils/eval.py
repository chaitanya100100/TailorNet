class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate():
    """Evaluate TailorNet (or any model for that matter) on test set."""
    from dataset.static_pose_shape_final import MultiStyleShape
    import torch
    from torch.utils.data import DataLoader
    from utils.eval import AverageMeter
    from models import ops

    gender = 'male'
    garment_class = 'short-pant'

    dataset = MultiStyleShape(garment_class=garment_class, gender=gender, split='test')
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False, drop_last=False)
    print(len(dataset))

    val_dist = AverageMeter()
    from models.tailornet_model import get_best_runner as tn_runner
    runner = tn_runner(garment_class, gender)
    # from trainer.base_trainer import get_best_runner as baseline_runner
    # runner = baseline_runner("/BS/cpatel/work/data/learn_anim/{}_{}_weights/tn_orig_baseline/{}_{}".format(garment_class, gender, garment_class, gender))

    device = torch.device('cuda:0')
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            gt_verts, thetas, betas, gammas, _ = inputs

            thetas, betas, gammas = ops.mask_inputs(thetas, betas, gammas, garment_class)
            gt_verts = gt_verts.to(device)
            thetas = thetas.to(device)
            betas = betas.to(device)
            gammas = gammas.to(device)
            pred_verts = runner.forward(thetas=thetas, betas=betas, gammas=gammas).view(gt_verts.shape)

            dist = ops.verts_dist(gt_verts, pred_verts) * 1000.
            val_dist.update(dist.item(), gt_verts.shape[0])
            print(i, len(dataloader))
    print(val_dist.avg)


if __name__ == '__main__':
    evaluate()