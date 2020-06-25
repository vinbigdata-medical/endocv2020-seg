from yacs.config import CfgNode as CN


_C = CN()

_C.EXP = "" # Experiment name
_C.DEBUG = False

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 0
_C.SYSTEM.FP16 = True
_C.SYSTEM.OPT_L = "O2"
_C.SYSTEM.CUDA = True
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 8

_C.DIRS = CN()
_C.DIRS.DATA = "./data/"
_C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "./outputs/"
_C.DIRS.LOGS = "./logs/"

_C.DATA = CN()
_C.DATA.FOLD = 0
_C.DATA.AUGMENT = "randaug"
_C.DATA.RANDAUG = CN()
_C.DATA.RANDAUG.N = 3
_C.DATA.RANDAUG.M = 11
_C.DATA.AUGMIX = CN()
_C.DATA.AUGMIX.ALPHA = 1.
_C.DATA.AUGMIX.BETA = 1.

_C.DATA.CUTMIX = False
_C.DATA.MIXUP = False
_C.DATA.CM_ALPHA = 0.5
_C.DATA.IMG_SIZE = 512
_C.DATA.IN_CHANS = 3

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 40
_C.TRAIN.NUM_CYCLES = 4
_C.TRAIN.BATCH_SIZE = 8

_C.INFER = CN()
_C.INFER.TTA = False
_C.INFER.THRESHOLD = [0.5, 0.5, 0.5, 0.5, 0.5]

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.GD_STEPS = 1
_C.OPT.WARMUP_EPOCHS = 4
_C.OPT.BASE_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WEIGHT_DECAY_BIAS = 0.0
_C.OPT.EPSILON = 1e-3
_C.OPT.SWA = CN()
_C.OPT.SWA.START = 10
_C.OPT.SWA.FREQ = 5

_C.LOSS = CN()
_C.LOSS.NAME = "iou"
_C.LOSS.GAMMA = 2.
_C.LOSS.SMOOTH = 0.1
_C.LOSS.THRESH = 0.5

_C.MODEL = CN()
_C.MODEL.NAME = "FPNSegmentor"


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "resnet50"
_C.MODEL.BACKBONE.FUNC_NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.OUTPUT_STRIDE = 32
_C.MODEL.BACKBONE.PRETRAINED = True
# Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'.
_C.MODEL.BACKBONE.POOL_TYPE = "avg"
_C.MODEL.BACKBONE.DROPOUT = 0.2
_C.MODEL.BACKBONE.DROP_CONNECT = 0.2
# Name of the layers whose outputs should be returned in forward.
_C.MODEL.BACKBONE.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.BACKBONE.NUM_CLASSES = 1000


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
# Names of the input feature maps to be used by FPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.FPN.OUT_CHANNELS = 256
# Options: "" (no norm), "GN"
_C.MODEL.FPN.NORM = "BN"
# Options: "relu", "leaky_relu"
_C.MODEL.FPN.ACT = "relu"
# Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
_C.MODEL.FPN.FUSE_TYPE = "sum"


# ---------------------------------------------------------------------------- #
# Semantic Segmentation Head
# ---------------------------------------------------------------------------- #
_C.MODEL.SEM_SEG_HEAD = CN()
_C.MODEL.SEM_SEG_HEAD.NAME = "FPNHead"
_C.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
# Number of classes in the semantic segmentation head
_C.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 5
# Number of channels in the 3x3 convs inside semantic-FPN heads.
_C.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128
# Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
_C.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
# Normalization method for the convolution layers. Options: "" (no norm), "GN".
_C.MODEL.SEM_SEG_HEAD.NORM = "BN"
# Activation method for the convolution layers. Options: "relu", "leaky_relu".
_C.MODEL.SEM_SEG_HEAD.ACT = "relu"
_C.MODEL.SEM_SEG_HEAD.FUSE_TYPE = "cat"


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`