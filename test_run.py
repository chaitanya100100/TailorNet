import os
import numpy as np


ROOT = "/BS/cpatel/work/code_test"


def save_for_one_frame():
    import sys
    if sys.argv[1] not in ["generate", "save_image"]:
        print("specify either <generate> <save_image>")
        exit(-1)
    SAVE_IMG = (sys.argv[1] == "save_image")

    import torch
    from utils.interpenetration import remove_interpenetration_fast
    from models.smpl4garment import SMPL4Garment
    from psbody.mesh import Mesh
    from utils.rotation import normalize_y_rotation
    import numpy as np
    import os
    from visualization.utils import get_shape, get_style, get_specific_pose

    garment_class = 't-shirt'
    gender = 'male'
    shape_idx = '002'
    style_idx = '005'
    idxs = [32, 96, 112, 128]

    smpl = SMPL4Garment(gender=gender)

    if not SAVE_IMG:
        from models.tailornet_model import get_best_runner
        runner = get_best_runner(gender=gender, garment_class=garment_class)
        from trainer.base_trainer import get_best_runner as bl_runner
        runner_bl = bl_runner("/BS/cpatel/work/data/learn_anim/test_mlp_baseline/t-shirt_male")

        from dataset.static_pose_shape_final import OneStyleShape
        ds = OneStyleShape(garment_class=garment_class, gender=gender,
                           shape_idx=shape_idx, style_idx=style_idx, split=None)
        for id in idxs:
            gt_verts, theta, beta, gamma, _ = ds[id]
            theta = theta.numpy()
            beta = beta.numpy()
            gamma = gamma.numpy()

            theta_normalized = normalize_y_rotation(theta).astype(np.float32)

            theta_torch = torch.from_numpy(theta_normalized).unsqueeze(0).cuda()
            beta_torch = torch.from_numpy(beta).unsqueeze(0).cuda()
            gamma_torch = torch.from_numpy(gamma).unsqueeze(0).cuda()
            with torch.no_grad():
                pred_verts = runner.forward(theta_torch, beta_torch, gamma_torch)[0].cpu().view(-1, 3).numpy()
                pred_verts_bl = runner_bl.forward(
                    thetas=theta_torch, betas=beta_torch, gammas=gamma_torch)[0].cpu().view(-1, 3).numpy()
            body, gt_gar = smpl.run(beta=beta, theta=theta_normalized, garment_class=garment_class, garment_d=gt_verts)
            body, pred_gar = smpl.run(beta=beta, theta=theta_normalized, garment_class=garment_class, garment_d=pred_verts)
            pred_gar_processed = remove_interpenetration_fast(pred_gar, body)
            body, pred_gar_bl = smpl.run(beta=beta, theta=theta_normalized, garment_class=garment_class, garment_d=pred_verts_bl)
            pred_gar_bl_processed = remove_interpenetration_fast(pred_gar_bl, body)

            body.write_ply(os.path.join(ROOT, "body_{}.ply".format(id)))
            gt_gar.write_ply(os.path.join(ROOT, "gt_gar_{}.ply".format(id)))
            # pred_gar.write_ply(os.path.join(ROOT, "pred_gar_{}.ply".format(id)))
            pred_gar_processed.write_ply(os.path.join(ROOT, "pred_gar_processed_{}.ply".format(id)))
            # pred_gar_bl.write_ply(os.path.join(ROOT, "pred_gar_bl_{}.ply".format(id)))
            pred_gar_bl_processed.write_ply(os.path.join(ROOT, "pred_gar_bl_processed_{}.ply".format(id)))

    else:
        for id in idxs:
            body = Mesh(filename=os.path.join(ROOT, "body_{}.ply".format(id)))
            gt_gar = Mesh(filename=os.path.join(ROOT, "gt_gar_{}.ply".format(id)))
            # pred_gar = Mesh(filename=os.path.join(ROOT, "pred_gar_{}.ply".format(id)))
            pred_gar_processed = Mesh(filename=os.path.join(ROOT, "pred_gar_processed_{}.ply".format(id)))
            # pred_gar_bl = Mesh(filename=os.path.join(ROOT, "pred_gar_bl_{}.ply".format(id)))
            pred_gar_bl_processed = Mesh(filename=os.path.join(ROOT, "pred_gar_bl_processed_{}.ply".format(id)))

            from visualization.blender_renderer import visualize_garment_body
            visualize_garment_body(pred_gar_processed, body,
                                   os.path.join(ROOT, "ours_front_{}.png".format(id)), side='front')
            # visualize_garment_body(pred_gar_processed, body, os.path.join(ROOT, "ours_back.png"), side='back')
            # visualize_garment_body(pred_gar_processed, body, os.path.join(ROOT, "ours_right90.png"), side='right90')
            visualize_garment_body(pred_gar_bl_processed, body,
                                   os.path.join(ROOT, "BL_front_{}.png".format(id)), side='front')
            # visualize_garment_body(pred_gar_bl_processed, body, os.path.join(ROOT, "BL_back.png"), side='back')
            # visualize_garment_body(pred_gar_bl_processed, body, os.path.join(ROOT, "BL_right90.png"), side='right90')
            visualize_garment_body(gt_gar, body,
                                   os.path.join(ROOT, "GT_front_{}.png".format(id)), side='front')
            # visualize_garment_body(gt_gar, body, os.path.join(ROOT, "GT_back.png"), side='back')
            # visualize_garment_body(gt_gar, body, os.path.join(ROOT, "GT_right90.png"), side='right90')


def crop_join():
    import cv2
    import numpy as np
    import os

    fnames = []
    for side in ["front"]:
        row = []
        for meth in ["BL", "ours", "GT"]:
            # if meth == "ours":continue
            row.append(os.path.join(ROOT, "{}_{}.png".format(meth, side)))
        fnames.append(row)

    outpath = os.path.join(ROOT, "check.png")

    def crop_it(img):
        return img[200:200 + 950, 620: 620 + 800]

    rows = []
    for one_row in fnames:
        this_row = []
        for img_path in one_row:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = crop_it(img)
            img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0,0,0,1))

            this_row.append(img)
        rows.append(np.hstack(this_row))

    final_img = np.vstack(rows)
    cv2.imwrite(outpath, final_img)


if __name__ == '__main__':
    import sys
    if sys.argv[1] == "crop_join":
        crop_join()
    else:
        save_for_one_frame()