from collections import namedtuple

PRIMITIVES = [
    "conv",
    "dil_conv_d2",
    "dil_conv_d3",
    "dil_conv_d4",
    "Dsep_conv",
    "invert_Bot_Conv_E2",
    "conv_transpose",
    "identity",
]

CHANNELS = [16, 32, 48, 64, 16, 32, 48, 64]
REPEAT = [1, 2, 3, 4, 1, 2, 3, 4]
KERNELS = [1, 3, 5, 7, 1, 3, 5, 7]

Genotype = namedtuple("Genotype", ["Branch1", "Branch2", "Branch3"])

DEFAULT_N_VAR = 84
DEFAULT_N_OBJ = 2
DEFAULT_POP_SIZE = 100
DEFAULT_N_GEN = 500
DEFAULT_UPSCALE = 2
DEFAULT_INPUT_CHANNELS = 3

DEFAULT_ENSEMBLE_METHOD = "mean"   # mean | median | weighted_mean