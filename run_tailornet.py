import os
import numpy as np
import torch
import time

from psbody.mesh import Mesh

from models.tailornet_model import get_best_runner as get_tn_runner
from models.smpl4garment import SMPL4Garment
from utils.rotation import normalize_y_rotation
from visualization.blender_renderer import visualize_garment_body

from dataset.canonical_pose_dataset import get_style, get_shape
from visualization.vis_utils import get_specific_pose, get_specific_style_old_tshirt
from visualization.vis_utils import get_specific_shape, get_amass_sequence_thetas
from utils.interpenetration import remove_interpenetration_fast

# Set output path where inference results will be stored
OUT_PATH = "/content/output"


def get_single_frame_inputs(garment_class, gender):
    """Prepare some individual frame inputs."""
    betas = [
        get_specific_shape('tallthin'),
        get_specific_shape('shortfat'),
        get_specific_shape('mean'),
        get_specific_shape('somethin'),
        get_specific_shape('somefat'),
    ]
    # old t-shirt style parameters are centered around [1.5, 0.5, 1.5, 0.0]
    # whereas all other garments styles are centered around [0, 0, 0, 0]
    if garment_class == 'old-t-shirt':
        gammas = [
            get_specific_style_old_tshirt('mean'),
            get_specific_style_old_tshirt('big'),
            get_specific_style_old_tshirt('small'),
            get_specific_style_old_tshirt('shortsleeve'),
            get_specific_style_old_tshirt('big_shortsleeve'),
        ]
    else:
        gammas = [
            get_style('000', garment_class=garment_class, gender=gender),
            get_style('001', garment_class=garment_class, gender=gender),
            get_style('002', garment_class=garment_class, gender=gender),
            get_style('003', garment_class=garment_class, gender=gender),
            get_style('004', garment_class=garment_class, gender=gender),
        ]
    thetas = [
        get_specific_pose(0),
        get_specific_pose(1),
        get_specific_pose(2),
        get_specific_pose(3),
        get_specific_pose(4),
    ]
    return thetas, betas, gammas


def get_sequence_inputs(garment_class, gender):
    """Prepare sequence inputs."""
    beta = get_specific_shape('somethin')
    if garment_class == 'old-t-shirt':
        gamma = get_specific_style_old_tshirt('big_longsleeve')
    else:
        gamma = get_style('000', gender=gender, garment_class=garment_class)

    # downsample sequence frames by 2
    thetas = get_amass_sequence_thetas('05_02')[::2]

    betas = np.tile(beta[None, :], [thetas.shape[0], 1])
    gammas = np.tile(gamma[None, :], [thetas.shape[0], 1])
    return thetas, betas, gammas


def run_tailornet():
    gender = 'female'
    garment_class = 'short-pant'
    thetas, betas, gammas = get_single_frame_inputs(garment_class, gender)
    # # uncomment the line below to run inference on sequence data
    # thetas, betas, gammas = get_sequence_inputs(garment_class, gender)

    # load model
    tn_runner = get_tn_runner(gender=gender, garment_class=garment_class)
    # from trainer.base_trainer import get_best_runner
    # tn_runner = get_best_runner("/BS/cpatel/work/data/learn_anim/tn_baseline/{}_{}/".format(garment_class, gender))
    smpl = SMPL4Garment(gender=gender)

    # make out directory if doesn't exist
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)

    # run inference
    for i, (theta, beta, gamma) in enumerate(zip(thetas, betas, gammas)):
        print(i, len(thetas))
        # normalize y-rotation to make it front facing
        theta_normalized = normalize_y_rotation(theta)
        with torch.no_grad():
            pred_verts_d = tn_runner.forward(
                thetas=torch.from_numpy(theta_normalized[None, :].astype(np.float32)).cuda(),
                betas=torch.from_numpy(beta[None, :].astype(np.float32)).cuda(),
                gammas=torch.from_numpy(gamma[None, :].astype(np.float32)).cuda(),
            )[0].cpu().numpy()

        # get garment from predicted displacements
        body, pred_gar = smpl.run(beta=beta, theta=theta, garment_class=garment_class, garment_d=pred_verts_d)
        pred_gar = remove_interpenetration_fast(pred_gar, body)

        # color the verticies
        color_map(pred_gar,body);

        # save body and predicted garment
        body.write_ply(os.path.join(OUT_PATH, "body_{:04d}.ply".format(i)))
        pred_gar.write_ply(os.path.join(OUT_PATH, "pred_gar_{:04d}.ply".format(i)))

def color_map(pred_gar, body):
    t = np.arange(len(pred_gar.v)).reshape(len(pred_gar.v),1)
    a = np.append(pred_gar.v, t, axis = 1)

    pred_gar_sorted = a[a[:,1].argsort()]

    ssss = time.time()
    start  = time.time()
    closest = np.asarray(body.closest_vertices(pred_gar.v)[1]) # (verts , dist)
    print("closest time = ",time.time() - start)

    start = time.time()

    n = len(pred_gar.v)
    m = n/10

    for x in range(10):
      mean_dist = 0
      for j in range(int(m)):
        mean_dist = mean_dist + closest[int(pred_gar_sorted[x*int(m) + j][-1])]
      mean_dist = mean_dist/m
      for j in range(int(m)):
        closest[int(pred_gar_sorted[x*int(m)+j][-1])] = mean_dist

    # # blend
    # for x in range(9):
    #   mean_dist = 0
    #   for j in range(int(m)):
    #     mean_dist = mean_dist + closest[int(pred_gar_sorted[x*int(m) + j + int(m/2)][-1])]
    #   mean_dist = mean_dist/m
    #   for j in range(int(m)):
    #     closest[int(pred_gar_sorted[x*int(m)+j+int(m/2)][-1])] = mean_dist

    print("after dist mean change time = ",time.time() - start)
    start = time.time()

    pred_gar.set_vertex_colors(np.array([0,1,0]))

    tmp = np.asarray(closest)
    g = (tmp > .0075)
    r = (tmp < .0091)
    print(np.min(tmp))
    print(np.max(tmp))
    for j in range(len(closest)):
      pred_gar.set_vertex_colors(np.array([r[j], g[j], 0]),j)
    print("after coloring time = ", time .time() - start)

    print("all time  = " ,time.time()-ssss)

def render_images():
    """Render garment and body using blender."""
    i = 0
    while True:
        body_path = os.path.join(OUT_PATH, "body_{:04d}.ply".format(i))
        if not os.path.exists(body_path):
            break
        body = Mesh(filename=body_path)
        pred_gar = Mesh(filename=os.path.join(OUT_PATH, "pred_gar_{:04d}.ply".format(i)))

        visualize_garment_body(
            pred_gar, body, os.path.join(OUT_PATH, "img_{:04d}.png".format(i)), garment_class='t-shirt', side='front')
        i += 1

    # Concate frames of sequence data using this command
    # ffmpeg -r 10 -i img_%04d.png -vcodec libx264 -crf 10  -pix_fmt yuv420p check.mp4
    # Make GIF
    # convert -delay 200 -loop 0 -dispose 2 *.png check.gif
    # convert check.gif -resize 512x512 check_small.gif


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1 or sys.argv[1] == 'inference':
        run_tailornet()
    elif sys.argv[1] == 'render':
        render_images()
    else:
        raise AttributeError
