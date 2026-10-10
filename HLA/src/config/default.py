from yacs.config import CfgNode as CN
_CN = CN()

##############  ↓  HLA Pipeline  ↓  ##############
_CN.HLA = CN()
_CN.HLA.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.HLA.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd

_CN.HLA.COARSE = CN()
_CN.HLA.COARSE.D_MODEL = 128

_CN.HLA.MATCH_COARSE = CN()
_CN.HLA.MATCH_COARSE.USE_SM = True
_CN.HLA.MATCH_COARSE.THR = 0.2
_CN.HLA.MATCH_COARSE.BORDER_RM = 2
_CN.HLA.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.HLA.MATCH_COARSE.SKH_ITERS = 3
_CN.HLA.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.HLA.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.HLA.MATCH_COARSE.INFERENCE = False

_CN.HLA.FINE = CN()
_CN.HLA.FINE.D_MODEL = 32
_CN.HLA.FINE.DENSER = False # if true, match all features in fine-level windows
_CN.HLA.FINE.INFERENCE = False
_CN.HLA.FINE.DSMAX_TEMPERATURE = 0.1
_CN.HLA.FINE.THR = 0.1

_CN.HLA.LOSS = CN()
_CN.HLA.LOSS.COARSE_WEIGHT = 1.0
_CN.HLA.LOSS.FOCAL_ALPHA = 0.25
_CN.HLA.LOSS.FOCAL_GAMMA = 2.0
_CN.HLA.LOSS.POS_WEIGHT = 1.0
_CN.HLA.LOSS.NEG_WEIGHT = 1.0
_CN.HLA.LOSS.SUB_WEIGHT = 1 * 10**4
_CN.HLA.LOSS.FINE_WEIGHT = 1.0

_CN.HLA.MP = False  # just for reproducing paper, FP16 is much faster on modern GPUs
_CN.HLA.EVAL_TIMES = 1

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None
_CN.DATASET.INPUT_PAIRS = None
_CN.DATASET.NPE_NAME = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']
_CN.DATASET.CORR_TH = 5

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 5
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.EN_LR = None
_CN.TRAINER.DE_LR = None
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 1e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'LO-RANSAC'
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5


from src.config.default import _CN as cfg          
from yacs.config import CfgNode as CN             


cfg.HLA.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
cfg.HLA.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
cfg.HLA.COARSE.D_MODEL = 256
cfg.HLA.FINE.D_MODEL = 64

cfg.TRAINER.SCHEDULER = 'CosineAnnealing'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
cfg.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
cfg.TRAINER.CANONICAL_BS = 2
cfg.TRAINER.CANONICAL_LR = 1e-4
cfg.TRAINER.WARMUP_STEP = 58200  # 3 epochs

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.HLA.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
cfg.HLA.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 20
cfg.TRAINER.N_VAL_PAIRS_TO_PLOT = 1

cfg.TRAINER.SEED = 66









def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
