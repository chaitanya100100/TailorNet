import os
import numpy as np
import torch
import time

from models.tailornet_model import get_best_runner as get_tn_runner
from models.smpl4garment import SMPL4Garment
from utils.rotation import normalize_y_rotation

# from dataset.canonical_pose_dataset import get_style, get_shape
from dataset.canonical_pose_dataset import get_style
from utils.interpenetration import remove_interpenetration_fast
from visualization.vis_utils import get_specific_pose
from visualization.vis_utils import get_specific_shape

# Set output path where inference results will be stored
OUT_PATH = "/content/output"

def gen_body(thetas=get_specific_pose(0),betas=get_specific_shape('mean')):
    smpl = SMPL4Garment(gender=gender)
    body,_ = smpl.run(beta=beta, theta=theta)
    body.write_obj("../models/obj/body.obj")

def gen_body_gar(thetas=get_specific_pose(0), betas=get_specific_shape('mean'), gender='female', garment_class='short-pant'):
    gammas = get_style('000', garment_class=garment_class, gender=gender)

    # load model
    tn_runner = get_tn_runner(gender=gender, garment_class=garment_class)

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

        # pred_gar.set_texture_image('/content/TailorNet/tex.jpg') 

        # save body and predicted garment
        # body.write_ply(os.path.join(OUT_PATH, "body_{:04d}.ply".format(i)))
        body.write_ply("../models/body.ply")
        body.write_obj("../models/obj/body.obj")
        # pred_gar.write_ply(os.path.join(OUT_PATH, "pred_gar_{:04d}.ply".format(i)))
        pred_gar.write_ply("../models/gar.ply")
        pred_gar.write_obj("../models/obj/gar.obj")
    os.system('python texture_mesh.py "../models/obj/gar.obj" "../models/tex/gar.jpg"')

def color_map(pred_gar, body):
    t = np.arange(len(pred_gar.v)).reshape(len(pred_gar.v),1)

    #TODO: sort and return the order only no need to swap the lists
    # and no need to append the array
    a = np.append(pred_gar.v, t, axis = 1)
    pred_gar_sorted = a[a[:,1].argsort()]

    ssss = time.time()
    start  = time.time()
    closest = np.asarray(body.closest_vertices(pred_gar.v)[1]) # (verts , dist)
    print("closest time = ",time.time() - start)

    start = time.time()

    n = len(pred_gar.v)
    m = n/10

    #TODO: change 10 to a number that the vertices can divide on with no reminder
    for x in range(10):
      mean_dist = 0
      for j in range(int(m)):
        mean_dist = mean_dist + closest[int(pred_gar_sorted[x*int(m) + j][-1])]
      mean_dist = mean_dist/m
      for j in range(int(m)):
        closest[int(pred_gar_sorted[x*int(m)+j][-1])] = mean_dist

    #TODO:blend ( using gaussian function ? )
    # for x in range(9):
    #   mean_dist = 0
    #   for j in range(int(m)):
    #     mean_dist = mean_dist + closest[int(pred_gar_sorted[x*int(m) + j + int(m/2)][-1])]
    #   mean_dist = mean_dist/m
    #   for j in range(int(m)):
    #     closest[int(pred_gar_sorted[x*int(m)+j+int(m/2)][-1])] = mean_dist

    print("after dist mean change time = ",time.time() - start)
    start = time.time()

    #TODO: not supposed to call this one 
    pred_gar.set_vertex_colors(np.array([0,0,0]))

    
    #TODO: change to more meaningfull threasholds
    g = (closest > .0075)
    r = (closest < .0091)

    print(np.min(closest))
    print(np.max(closest))

    #TODO: send the whole vector ?
    for j in range(len(closest)):
      pred_gar.set_vertex_colors(np.array([r[j], g[j], 0]),j)

    print("after coloring time = ", time .time() - start)
    print("all time  = " ,time.time()-ssss)

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        gen_body()
    else:
        gen_body_gar(sys.argv[1],sys.argv[2],sys.argv[3] ,sys.argv[4])
