import onnx
from graph import Graph
from tensor import Tensor
from const import *

class InputNotFoundError(Exception):
    """Raised when cannot find input tensors """
    pass

# onnx tensor_dtype_to_np_dtype
# cite https://xadupre.github.io/draft/onnx/onnx_python/mapping.html
# def tensor_dtype_to_np_dtype(tensor_dtype: int) -> np.dtype:
#     return mapping.TENSOR_TYPE_MAP[tensor_dtype].np_dtype

def get_padding_mode(padding):
    if (padding == "SAME"):
        return PaddingMode.PD_MODE_SAME
    elif (padding == "VALID"):
        return PaddingMode.PD_MODE_VALID
    else:
        assert(False)

def get_activation_mode(activation):
    if (activation == "NONE"):
        return ActiMode.AC_MODE_NONE
    elif (activation == "SIGMOID"):
        return ActiMode.AC_MODE_SIGMOID
    elif (activation == "RELU"):
        return ActiMode.AC_MODE_RELU
    elif (activation == "TANH"):
        return ActiMode.AC_MODE_TANH
    else:
        assert(False)

def parse_attribute(attributes) -> dict:
    atts = dict()
    for att in attributes:
        if att.type == onnx.AttributeProto.INT:
            atts[att.name] = att.i
        elif att.type == onnx.AttributeProto.INTS:
            atts[att.name] = att.ints
        elif att.type == onnx.AttributeProto.FLOAT:
            atts[att.name] = att.f
        elif att.type == onnx.AttributeProto.STRING:
            atts[att.name] = att.s
        elif att.type == onnx.AttributeProto.TENSOR:
            atts[att.name] = att.t
        else:
            assert False, "Unsupported Attribute Type: {}".format(att.type)
    return atts

def get_conv_pool_pads_attr(attrs) -> str:
    if ("auto_pad" in attrs):
        padding = attrs["auto_pad"]
        if isinstance(padding, bytes):
            padding = padding.decode()
        if (padding=='SAME_LOWER') or (padding=='SAME_UPPER'):
            pads = "SAME"
        elif (padding=='VALID'):
            pads = "VALID"
        else:
            assert padding=='NOTSET', "Unrecogonized auto_pad value: {}".format(padding)
        # Note that we always think conv1x1 has SAME padding
        # This will allow fusing enlarged convs
        if sum(attrs['kernel_shape']) <= 2:
            pads = "SAME"
        if padding != 'NOTSET':
            return pads
    # Assume zero padding if the pads are missing
    if "pads" not in attrs:
        attrs['pads'] = [0 for i in range(len(attrs['kernel_shape'])*2)]
    # Note that we always think conv1x1 has SAME padding
    # This will allow fusing enlarged convs
    if sum(attrs["pads"]) == 0 and sum(attrs['kernel_shape']) > 2:
        pads = "VALID"
    else:
        pads = "SAME"
    return pads

def get_inputs(op: onnx.NodeProto, graph: Graph, tensors: list[Tensor], initializer) -> list:
    inputs = list()
    for i in op.input:
        input_tensor = None
        if i in tensors.keys():
            input_tensor = tensors[i]
        else:
            for init in initializer:
                if init.name == i:
                    input_tensor = graph.new_weight(
                        dims=list(init.dims), data=init)
                    break
        if input_tensor is None:
            raise InputNotFoundError
        inputs.append(input_tensor)
    return inputs

def permutation_to_index(perm: list[int]) -> int:
    # check perm
    for i in range(len(perm)):
        assert(perm[i] >= 0 and perm[i] < len(perm))
        for j in range(i+1, len(perm)):
            assert(perm[i] != perm[j])
    idx = 0
    for i in range(len(perm)):
        idx = idx * len(perm) + perm[i]
    return idx