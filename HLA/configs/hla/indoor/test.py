from src.config.default import _CN as cfg

# pose estimation
cfg.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)

cfg.HLA.MP = False
cfg.HLA.EVAL_TIMES = 1
cfg.HLA.MATCH_COARSE.INFERENCE = True
cfg.HLA.FINE.INFERENCE = True
cfg.HLA.MATCH_COARSE.USE_SM = True
cfg.HLA.MATCH_COARSE.THR = 0.2
cfg.HLA.FINE.THR = 0.1

cfg.HLA.MATCH_COARSE.BORDER_RM = 2
