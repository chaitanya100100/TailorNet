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
[chumpy](https://github.com/mattloper/chumpy)  
opencv-python  
cython  
[psbody.mesh](https://github.com/MPI-IS/mesh)

## How to Run
- Download and prepare SMPL model and TailorNet data from [here](https://github.com/zycliao/TailorNet_dataset).
- Set DATA_DIR and LOG_DIR variables in `global_var.py` file.
- Download trained models from here. [Coming Soon]
- Run TailorNet inference as `python run_tailornet.py`.

## Training TailorNet yourself
- Set appropriate global variables in `global_var.py`, especially LOG_DIR.

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