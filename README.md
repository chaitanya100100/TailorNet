[Under Development]

# TailorNet Training and Models
This repository contains training code for "TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style" (CVPR 2020 Oral)  

[[arxiv](https://arxiv.org/abs/2003.04583)]
[[project website](https://virtualhumans.mpi-inf.mpg.de/tailornet/)]
[[Dataset Repo](https://github.com/zycliao/TailorNet_dataset)]
[[Youtube](https://www.youtube.com/watch?v=F0O21a_fsBQ)]

## Requirements
python3  
pytorch  
scipy  
[chumpy](https://github.com/mattloper/chumpy)  
[psbody.mesh](https://github.com/MPI-IS/mesh)

## How to Run
- Download and prepare SMPL model and TailorNet data from [here](https://github.com/zycliao/TailorNet_dataset).
- Set DATA_DIR variable in `global_var.py` file.
- Download trained models from here. [Coming Soon]
- Set output path in `run_tailornet.py` and run it to predict garments on some random inputs. You can play with 
  different inputs. You can also run inference on motion sequence data.
- To visualize predicted garment using blender, run `python run_tailornet.py render`. (blender needs to be installed.)

## Training TailorNet yourself
- Set appropriate global variables in `global_var.py`, especially LOG_DIR where training logs will be stored.
- Set appropriate config variables in `trainer/base_trainer.py` and run `python trainer/base_trainer.py` to train
simple MLP baseline.
- Similarly, run `trainer/lf_trainer.py` for training low frequency predictor and `trainer/ss2g_trainer.py` for
shape-style-to-garment(in canonical pose) model.
- Run `python trainer/hf_trainer.py --shape_style <shape1>_<style1> <shape2>_<style2>` to train pivot high frequency
predictors for pivots `<shape1>_<style1>` and `<shape2>_<style2>`. See `DATA_DIR/<garment_class>_<gender>/pivots.txt`
to know available pivots.
- Use `models.tailornet_model.TailorNetModel` to do prediction.

## Citation
Cite us:
```
@inproceedings{patel20tailornet,
        title = {TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style},
        author = {Patel, Chaitanya and Liao, Zhouyingcheng and Pons-Moll, Gerard},
        booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {jun},
        organization = {{IEEE}},
        year = {2020},
    }
```

### Misc
- `smpl_lib` follows MultiGarmentNet repo's [lib](https://github.com/bharat-b7/MultiGarmentNetwork/tree/master/lib).
Thanks Bharat for that.
- Thanks to Garvita for helping out during data generation procedure.