import numpy as np
import torch

from models.tailornet_model import get_best_runner as get_tn_runner
from models.smpl4garment import SMPL4Garment

from visualization.vis_utils import get_specific_pose, get_specific_style, \
    get_specific_shape, get_amass_sequence_thetas


def run_one_inference():
    gender = 'male'
    garment_class = 't-shirt'

    # set pose, shape and style
    beta = get_specific_shape('tallthin')
    gamma = get_specific_style('mean')
    theta = get_specific_pose(2)

    # load model
    tn_runner = get_tn_runner(gender=gender, garment_class=garment_class)
    smpl = SMPL4Garment(gender=gender)

    # run inference
    with torch.no_grad():
        pred_verts_d = tn_runner.forward(
            thetas=torch.from_numpy(theta[None, :].astype(np.float32)).cuda(),
            betas=torch.from_numpy(beta[None, :].astype(np.float32)).cuda(),
            gammas=torch.from_numpy(gamma[None, :].astype(np.float32)).cuda(),
        )[0].cpu().numpy()
    print(pred_verts_d)
    body, pred_gar = smpl.run(beta=beta, theta=theta, garment_class=garment_class, garment_d=pred_verts_d)

    # save body and predicted garment
    body.write_ply("/BS/cpatel/work/body.ply")
    pred_gar.write_ply("/BS/cpatel/work/pred_gar.ply")


if __name__ == '__main__':
    run_one_inference()