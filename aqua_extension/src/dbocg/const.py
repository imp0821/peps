from enum import Enum, auto

MAX_DIM                     =   8
MAX_NUM_SPLITS              =   32
MAX_NUM_INPUTS              =   6
MAX_NUM_OUTPUTS             =   6
MAX_TENSOR_SIZE             =   512 * 1024 * 1024 # 512MB
REPEAT_TIMES                =   32
WARMUP_TIMES                =   8

 
GUID_INVALID = 0
GUID_INPUT = 10
GUID_WEIGHT = 11
GUID_PRESERVED = 19


class DataType(Enum):
    DT_FLOAT  = 111
    DT_DOUBLE = 222
    DT_HALF   = 333
    DT_INT8   = 444
    DT_UINT8  = 555
    DT_INT32  = 666
    DT_INT64  = 777
    DT_BOOL   = 888


class OpType(Enum):
    OP_INPUT = 0
    OP_WEIGHT = 1
    OP_ANY = 2
    OP_CONV2D = 3
    OP_DROPOUT = 4
    OP_LINEAR = 5
    OP_POOL2D_MAX = 6
    OP_POOL2D_AVG = 7
    OP_RELU = 8
    OP_SIGMOID = 9
    OP_TANH = 10
    OP_BATCHNORM = 11
    OP_CONCAT = 12
    OP_SPLIT = 13
    OP_RESHAPE = 14
    OP_TRANSPOSE = 15
    OP_EW_ADD = 16
    OP_EW_MUL = 17
    OP_MATMUL = 18
    OP_MUL = 19
    OP_ENLARGE = 20
    OP_MERGE_GCONV = 21
    OP_CONSTANT_IMM = 22
    OP_CONSTANT_ICONV = 23
    OP_CONSTANT_ONE = 24
    OP_CONSTANT_POOL = 25
    OP_SQUEEZE = 26
    OP_UNSQUEEZE = 27
    OP_EW_SUB = 28
    OP_EW_DIV = 29
    OP_EW_EQUAL = 30
    OP_EW_GREATER = 31
    OP_EW_LESS = 32
    OP_EW_MAX = 33
    OP_EW_MIN = 34
    OP_REDUCE_ARGMAX = 35
    OP_REDUCE_ARGMIN = 36
    OP_REDUCE_MAX = 37
    OP_REDUCE_MEAN = 38
    OP_REDUCE_MIN = 39
    OP_REDUCE_PROD = 40
    OP_REDUCE_SUM = 41
    OP_PAD = 42
    OP_SHAPE = 43
    OP_SIZE = 44
    OP_TOPK = 45
    OP_WHERE = 46
    OP_CEIL = 47
    OP_CAST = 48
    OP_EXP = 49
    OP_ROUND = 50
    OP_LOG = 51
    OP_LOGICAL_NOT = 52
    OP_SQRT = 53
    OP_LEAKYRELU = 54
    OP_SLICE = 55
    OP_RESIZE = 56
    OP_PRELU = 57
    OP_FUSE_CONV_BATCHNORM = 58
    OP_FUSE_CONV_BATCHNORM_ALPHA_VAR = 59
    OP_FUSE_CONV_BATCHNORM_BIAS = 60
    OP_BROADCAST_ADD = 61


class PMParameter(Enum):
    PM_OP_TYPE = 0   	        # AnyOp
    PM_NUM_INPUTS = 1	        # AnyOp
    PM_NUM_OUTPUTS = 2	        # AnyOp
    PM_GROUP = 3                # Conv2D
    PM_KERNEL_H = 4		        # Conv2D, Pool2D
    PM_KERNEL_W = 5		        # Conv2D, Pool2D
    PM_STRIDE_H = 6		        # Conv2D, Pool2D
    PM_STRIDE_W = 7		        # Conv2D, Pool2D
    PM_PAD = 8		            # Conv2D, Pool2D
    PM_ACTI = 9		            # Conv2D, Pool2D
    PM_NUMDIM = 10		        # Concat, Transpose
    PM_AXIS = 11		        # Concat, Split
    PM_PERM = 12		        # Transpose
    PM_OUTSHUFFLE = 13	        # Transpose
    PM_MERGE_GCONV_COUNT = 14   # MergeGConv
    PM_AXES = 15		        # Squeeze, Unsqueeze, Reduce*
    PM_KEEP_DIMS = 16           # Reduce*
    PM_EPSILON = 17             # BatchNorm


class TNParameter(Enum):
    IN_0 = 100
    IN_1 = 101
    IN_2 = 102
    IN_3 = 103
    IN_4 = 104
    IN_5 = 105
    OU_0 = 200
    OU_1 = 201
    OU_2 = 202
    OU_3 = 203
    OU_4 = 204
    OU_5 = 205


class DIMParameter(Enum):
    DIM_0 = 300
    DIM_1 = 301
    DIM_2 = 302
    DIM_3 = 303
    DIM_ND = 310


class ActiMode(Enum):
    AC_MODE_NONE = 0
    AC_MODE_SIGMOID = 1
    AC_MODE_RELU = 2
    AC_MODE_TANH = 3


class PaddingMode(Enum):
    PD_MODE_SAME = 0
    PD_MODE_VALID = 1

# class ConstantMode(Enum):
#     CN_MODE_IDENTITY = 0
#     CN_MODE_ZEROS = 2
#     CN_MODE_ONES = 3
#     CN_MODE_ONES_SCALED_L1 = 4
#     CN_MODE_ONES_SCALED_L2 = 5
#     CN_MODE_ONES_SCALED_ALL = 6
    
# Construct operator table
op_table = dict()
op_table[OpType.OP_INPUT] = "Input"
op_table[OpType.OP_WEIGHT] = "Weight"
op_table[OpType.OP_CONV2D] = "Conv"
op_table[OpType.OP_DROPOUT] = "Dropout"
op_table[OpType.OP_POOL2D_MAX] = "MaxPool"
op_table[OpType.OP_POOL2D_AVG] = "AveragePool"
op_table[OpType.OP_RELU] = "Relu"
op_table[OpType.OP_SIGMOID] = "Sigmoid"
op_table[OpType.OP_TANH] = "Tanh"
op_table[OpType.OP_BATCHNORM] = "BatchNormalization"
op_table[OpType.OP_CONCAT] = "Concat"
op_table[OpType.OP_SPLIT] = "Split"
op_table[OpType.OP_RESHAPE] = "Reshape"
op_table[OpType.OP_TRANSPOSE] = "Transpose"
op_table[OpType.OP_EW_ADD] = "Add"
op_table[OpType.OP_EW_MUL] = "Mul"
op_table[OpType.OP_MATMUL] = "MatMul"
op_table[OpType.OP_SQUEEZE] = "Squeeze"
op_table[OpType.OP_UNSQUEEZE] = "Unsqueeze"
op_table[OpType.OP_EW_SUB] = "Sub"
op_table[OpType.OP_EW_DIV] = "Div"
op_table[OpType.OP_EW_EQUAL] = "Equal"
op_table[OpType.OP_EW_GREATER] = "Greater"
op_table[OpType.OP_EW_LESS] = "Less"
op_table[OpType.OP_EW_MAX] = "Max"
op_table[OpType.OP_EW_MIN] = "Min"
op_table[OpType.OP_REDUCE_ARGMAX] = "ArgMax"
op_table[OpType.OP_REDUCE_ARGMIN] = "ArgMin"
op_table[OpType.OP_REDUCE_MAX] = "ReduceMax"
op_table[OpType.OP_REDUCE_MEAN] = "ReduceMean"
op_table[OpType.OP_REDUCE_MIN] = "ReduceMin"
op_table[OpType.OP_REDUCE_PROD] = "ReduceProd"
op_table[OpType.OP_REDUCE_SUM] = "ReduceSum"
op_table[OpType.OP_PAD] = "Pad"
op_table[OpType.OP_SHAPE] = "Shape"
op_table[OpType.OP_SIZE] = "Size"
op_table[OpType.OP_TOPK] = "TopK"
op_table[OpType.OP_WHERE] = "Where"
op_table[OpType.OP_CEIL] = "Ceil"
op_table[OpType.OP_CAST] = "Cast"
op_table[OpType.OP_EXP] = "Exp"
op_table[OpType.OP_ROUND] = "Round"
op_table[OpType.OP_LOG] = "Log"
op_table[OpType.OP_LOGICAL_NOT] = "Not"
op_table[OpType.OP_SQRT] = "Sqrt"
op_table[OpType.OP_SLICE] = "Slice"
op_table[OpType.OP_RESIZE] = "Resize"
# op_table[OP_BROADCAST_ADD] = "BroadcastAdd"
op_table[OpType.OP_BROADCAST_ADD] = "Add"