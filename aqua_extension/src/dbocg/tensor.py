from const import *
from copy import deepcopy
from typing import List, Optional, Union
import onnx_tool
import onnx
import numpy

DATATYPE = float

class SplitInfo:
    NO_SPLIT = None   
    
    def __init__(self) -> None:
        self.num = 0
        self.pos = [0 for _ in range(MAX_NUM_SPLITS)]

    def __eq__(self, rhs: 'SplitInfo'):
        if self.num != rhs.num:
            return False
        for i in range(self.num):
            if self.pos[i] != rhs.pos[i]:
                return False
        return True

    def __ne__(self, rhs: 'SplitInfo'):
        if self.num != rhs.num:
            return True
        for i in range(self.num):
            if self.pos[i] != rhs.pos[i]:
                return True
        return False

    def merge(self, offset: int, next: 'SplitInfo') -> None:
        assert(self.num + 1 + next.num < MAX_NUM_SPLITS)
        for i in range(next.num):
            self.pos[self.num] = offset + next.pos[i]
            self.num += 1
        self.pos[self.num] = offset
        self.num += 1

    def divide(self, left: 'SplitInfo', right: 'SplitInfo'):
        assert(self.num > 0)
        left.num = 0
        right.num = 0
        mid = self.pos[self.num - 1]
        idx = 0
        while idx < self.num and self.pos[idx] < mid:
            left.pos[left.num] = self.pos[idx]
            left.num += 1
            idx += 1
        while idx < self.num - 1:
            right.pos[right.num] = self.pos[idx] - mid
            right.num += 1
            idx += 1
        return mid

    def combine(self, next: 'SplitInfo'):
        if self.num != next.num:
            self.num = 0
        for i in range(self.num):
            if self.pos[i] != next.pos[i]:
                self.num = 0
                return

    def serialize(self, keys, idx) -> None:
        keys[idx] = self.num
        idx += 1
        for i in range(self.num):
            keys[idx] = self.pos[i]
            idx += 1
    
    def copy_from(self, rhs):
        self.num = rhs.num
        self.pos = rhs.pos[:]

SplitInfo.NO_SPLIT = SplitInfo()

 
class Tensor:
    """
    Tensor class.

    Attributes:
        idx (int): idx is used for Ops with multiple outputs (e.g., split)
    """
    MAX_KEY_LENGTH = (MAX_NUM_SPLITS + 2) * MAX_DIM + 2
    MAGIC_NUMBER = 23333
    def __init__(self, ndim: Optional[int] = None, 
                 dims: Optional[List[int]] = None,
                 guid: Optional[int] = None, 
                 data: Optional[Union[onnx.ValueInfoProto, onnx.TensorProto, list[float]]] = None,
                 ) -> None:
        from op import Op
        if ndim and dims and guid:
            self.numDim = ndim
            self.idx = 0
            self.op = Op(guid, None)
            self.data_ptr = None
            assert (guid != 'GUID_INVALID')  # Replace 'GUID_INVALID' with the actual invalid guid
            assert (ndim <= MAX_DIM)
            count = 1
            self.dim = [0]*MAX_DIM   
            self.stride = [0]*MAX_DIM   
            self.split = [SplitInfo() for _ in range(MAX_DIM) ]   
            for i in range(ndim - 1, -1, -1):
                self.dim[i] = dims[i]
                self.stride[i] = count
                count *= self.dim[i]
                self.split[i].copy_from(SplitInfo.NO_SPLIT)
            if isinstance(data, onnx.TensorProto):
                self.data_ptr = tensorproto2ndarray(data, dims)
            elif isinstance(data, onnx.ValueInfoProto):
                raise NotImplementedError
            elif isinstance(data, list):
                self.data_ptr = numpy.array(data, dtype=numpy.float32).reshape(dims)
            elif isinstance(data, numpy.ndarray):
                self.data_ptr = data
            else:
                print("[tensor] data=None")
                self.data_ptr = None
        else:
            self.numDim = 0
            self.idx = 0 
            self.op = Op()
            self.data_ptr = None # replace c data_ptr as numpy 
            self.dim = [0]*MAX_DIM 
            self.stride = [0]*MAX_DIM
            self.split = [SplitInfo() for _ in range(MAX_DIM) ]   
            for i in range(MAX_DIM):
                self.split[i].num = 0

    def copy_from(self, src):
        self.numDim = src.numDim
        self.dim = deepcopy(src.dim)
        self.stride = deepcopy(src.stride)
        self.split = deepcopy(src.split)
        self.idx = src.idx
        self.op = src.op
        self.data_ptr = src.data_ptr

    def volume(self):
        ret = 1
        for i in range(self.numDim):
            ret *= self.dim[i]
        return ret

    def to_string(self, name):
        name = name + "("
        for i in range(self.numDim):
            suffix = ")" if i == self.numDim -1 else " "
            name = name + str(self.dim[i]) + ":" + str(self.stride[i]) + suffix
        return name

    def serialize(self, keys, idx):
        keys[idx] = self.MAGIC_NUMBER
        idx += 1
        keys[idx] = self.numDim
        idx += 1
        for i in range(self.numDim):
            keys[idx] = self.dim[i]
            idx += 1
        for i in range(self.numDim):
            keys[idx] = self.stride[i]
            idx += 1
        for i in range(self.numDim):
            self.split[i].serialize(keys, idx)

    def has_same_shape_stride_split(self, tensor):
        if self.numDim != tensor.numDim:
            return False
        for i in range(self.numDim):
            if self.dim[i] != tensor.dim[i] or self.stride[i] != tensor.stride[i] or self.split[i] != tensor.split[i]:
                return False
        return True

    def default_layout(self):
        cnt = 1
        for i in range(self.numDim - 1, -1, -1):
            if self.stride[i] != cnt: return False
            cnt *= self.dim[i]
        return True

TensorHandle = Tensor

def onnxdtype2npdtype(data_type):
    if data_type == onnx.TensorProto.FLOAT:
        return numpy.float32
    if data_type == onnx.TensorProto.DOUBLE:
        return numpy.float64
    if data_type == onnx.TensorProto.FLOAT16:
        return numpy.float16
    if data_type == onnx.TensorProto.INT32:
        return numpy.int32
    if data_type == onnx.TensorProto.INT16:
        return numpy.int16
    if data_type == onnx.TensorProto.INT64:
        return numpy.int64
    if data_type == onnx.TensorProto.INT8:
        return numpy.int8
    if data_type == onnx.TensorProto.UINT8:
        return numpy.uint8
    if data_type == onnx.TensorProto.BOOL:
        return numpy.bool_
    if data_type == onnx.TensorProto.STRING:
        return numpy.string_

def tensorproto2ndarray(initial, shape):
    # shape = shape_of_initializer(initial)
    ndtype = onnxdtype2npdtype(initial.data_type)
    if initial.raw_data == b'':
        arr = numpy.zeros(shape, ndtype).reshape((-1))
        if ndtype == numpy.float32:
            arr = numpy.fromiter(initial.float_data, dtype=ndtype)

        elif ndtype == numpy.int32:
            arr = numpy.fromiter(initial.int32_data, dtype=ndtype)

        elif ndtype == numpy.float16:
            raw = list(initial.int32_data)
            raw = numpy.fromiter(raw, dtype=numpy.uint16)
            mem = raw.tobytes()
            arr = numpy.frombuffer(mem, dtype=numpy.float16).reshape(shape)

        elif ndtype == numpy.int64:
            arr = numpy.fromiter(initial.int64_data, dtype=ndtype)

        elif ndtype == numpy.float64:
            arr = numpy.fromiter(initial.double_data, dtype=ndtype)

        elif ndtype == numpy.string_:
            arr = numpy.array(initial.string_data, dtype=ndtype)
    else:
        arr = numpy.frombuffer(initial.raw_data, dtype=ndtype)
    # if len(shape):
    arr = arr.reshape(shape)
    return arr
            
if __name__ == "__main__":
    si = SplitInfo()