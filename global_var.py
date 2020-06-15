import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/BS/cloth-anim/static00/tailor_data'
LOG_DIR = '/BS/cpatel/work/data/learn_anim'
TEMP_DIR = '/BS/cloth-anim/nobackup/temp'

GENDERS = ['neutral', 'male', 'female']

# Split path
SPLIT_FILE = '/BS/cpatel/work/data/split_static_pose_shape.npz'

# Name of garment class info file
GAR_INFO_FILE = 'garment_class_info.pkl'

# Set your SMPL paths here
SMPL_PATH_NEUTRAL = '/BS/RVH/work/data/smpl_models/neutral/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
SMPL_PATH_MALE = '/BS/RVH/work/data/smpl_models/lrotmin/lbs_tj10smooth6_0fixed_normalized/male/model.pkl'
SMPL_PATH_FEMALE = '/BS/RVH/work/data/smpl_models/lrotmin/lbs_tj10smooth6_0fixed_normalized/female/model.pkl'

# Skirt template path
SKIRT_TEMPLATE = "/BS/cpatel/work/data/garment/Skirt/smooth_Skirt.ply"

# outdir for smooth data
SMOOTH_OUT_DIR = "/BS/cpatel/static00/tailor_data"
CACHED_SMOOTH = False

"""
## SMPL joint
ID  parent  name
0   -1      ass
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
VALID_THETA = {
    't-shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19],
    'shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    'pants': [0, 1, 2, 4, 5, 7, 8],
    'skirt' : [0, 1, 2, ],
}
