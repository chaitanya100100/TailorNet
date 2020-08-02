import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset root directory. Change it to point to downloaded data root directory.
DATA_DIR = '/BS/cloth-anim/static00/tailor_data'

# Set the paths to SMPL model
SMPL_PATH_NEUTRAL = '/BS/RVH/work/data/smpl_models/neutral/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
SMPL_PATH_MALE = '/BS/RVH/work/data/smpl_models/lrotmin/lbs_tj10smooth6_0fixed_normalized/male/model.pkl'
SMPL_PATH_FEMALE = '/BS/RVH/work/data/smpl_models/lrotmin/lbs_tj10smooth6_0fixed_normalized/female/model.pkl'

# Log directory where training logs, checkpoints and visualizations will be stored
LOG_DIR = '/BS/cpatel/work/data/learn_anim'

# Downloaded TailorNet trained models' path
MODEL_WEIGHTS_PATH = "/BS/cpatel/work/data/learn_anim"

# --------------------------------------------------------------------
# Variable below hardly need to change
# --------------------------------------------------------------------

# Available genders
GENDERS = ['neutral', 'male', 'female']

# This file in DATA_DIR contains pose indices (out of all SMPL poses) of
# train/test splits as a dict {'train': <train_indices>, 'test': <test_indices>}
POSE_SPLIT_FILE = 'split_static_pose_shape.npz'

# This file in DATA_DIR contains garment template information in format
# { <garment_class>: {'vert_indices': <vert_indices>, 'f': <faces>} }
# where <vert_indices> refers to the indices of high_resolution SMPL
# template which make <garment_class> garment
GAR_INFO_FILE = 'garment_class_info.pkl'

# # Skirt template path
# SKIRT_TEMPLATE = "/BS/cpatel/work/data/garment/Skirt/smooth_Skirt.ply"

# Root dir for smooth data. Groundtruth smooth data is stored in the same
# data hierarchy as simulation data under this directory.
SMOOTH_DATA_DIR = DATA_DIR
# Indicates that smooth groundtruth data is available or not. If False, smoothing
# will be performed during the training which might slow down the training significantly.
SMOOTH_STORED = True

"""
## SMPL joint
ID  parent  name
0   -1      pelvis
1   0       L hip
2   0       R hip
3   0       stomach
4   1       L knee
5   2       R knee
6   3       Lower chest
7   4       L ankle
8   5       R ankle
9   6       Upper chest
10  7       L toe
11  8       R toe
12  9       throat
13  9       L Breast
14  9       R Breast
15  12      jaw
16  13      L shoulder
17  14      R shoulder
18  16      L elbow
19  17      R elbow
20  18      L wrist
21  19      R wrist
22  20      L hand
23  21      R hand
"""

# Lists the indices of joints which affect the deformations of particular garment
VALID_THETA = {
    't-shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19],
    'old-t-shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19],
    'shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    'pant': [0, 1, 2, 4, 5, 7, 8],
    'skirt' : [0, 1, 2, ],
}
