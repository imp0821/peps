from const import *
from tensor import Tensor
from op import *
from sortedcontainers import SortedDict
import logging
from typing import List

class KeyCompare:
    KEY_LENGTH = 0

    def __init__(self) -> None:
        pass

    def __lt__(self, other) ->bool: 
        if not isinstance(other, self.__class__):
            return False
        for i in range(self.KEY_LENGTH):
            if self.keys[i] != other.keys[i]:
                return self.keys[i] < other.keys[i]
            return False
    
    def __eq__(self, __value) -> bool:
        if not isinstance(__value, self.__class__):
            return False
        for i in range(self.KEY_LENGTH):
            if self.keys[i] != __value.keys[i]:
                return False
        return True
    
    def __hash__(self) -> int:
        return hash(tuple(self.keys[:self.KEY_LENGTH]))
    
# activation.cc
class ActivationKey(KeyCompare):
    KEY_LENGTH = Tensor.MAX_KEY_LENGTH + 2
    
    def __init__(self, _input: Tensor, _type: OpType, _inPlace: bool) -> None:
        super().__init__()
        self.keys = [0]*self.KEY_LENGTH
        idx = 0
        self.keys[idx] = _type
        idx += 1
        self.keys[idx] = int(_inPlace)
        idx += 1
        _input.serialize(self.keys, idx)


# element.cc
class ElementKey(KeyCompare):
    KEY_LENGTH = Tensor.MAX_KEY_LENGTH * 2 + 1

    def __init__(self, t1: Tensor, t2: Tensor, type: OpType) -> None:
        self.keys = [0]*self.KEY_LENGTH
        idx = 0
        self.keys[idx] = type
        idx += 1
        t1.serialize(self.keys, idx)
        t2.serialize(self.keys, idx)


# concat.cc
class ConcatKey(KeyCompare):
    KEY_LENGTH = MAX_NUM_INPUTS * Tensor.MAX_KEY_LENGTH + 3
    
    def __init__(self, axis: int, n: int,
                 _inputs: List[Tensor], _needCopy: List[bool]) -> None:
        self.keys = [0]*self.KEY_LENGTH
        idx = 0
        self.keys[:3] = [axis, n, self.bitmask(_needCopy)]
        idx += 3
        for i in range(n):
            _inputs[i].serialize(self.keys, idx)
    
    def bitmask(n: int, bits: List[bool]) -> int:
        ret = 0
        for i in range(n):
            ret = ret * 2 + 1 if bits[i] else ret * 2
        return ret

# conv2d.cc
class Conv2DKey(KeyCompare):
    KEY_LENGTH = Tensor.MAX_KEY_LENGTH * 2 + 4

    def __init__(self, _input: Tensor, _weight: Tensor, 
                 _strideH: int, _strideW:int, 
                 _padding: PaddingMode, _activation: ActiMode) -> None:
        self.keys = [0]*self.KEY_LENGTH
        assert _input.dim[1] % _weight.dim[1] == 0
        groups = _input.dim[1] // _weight.dim[1]
        assert _weight.dim[0] % groups == 0
        idx = 0
        self.keys[: 4] = [_strideH, _strideW, _padding, _activation]
        idx += 4
        _input.serialize(self.keys, idx)
        _weight.serialize(self.keys, idx)


# matmul.cc
class MatmulKey(KeyCompare):
    KEY_LENGTH = Tensor.MAX_KEY_LENGTH * 2 + 1
    def __init__(self, _input: Tensor, _weight: Tensor, _mode: ActiMode) -> None:
        super().__init__()
        self.keys = [0]*self.KEY_LENGTH
        assert _input.numDim == _weight.numDim
        idx = 0
        self.keys[idx] = _mode.value
        idx += 1
        _input.serialize(self.keys, idx)
        _weight.serialize(self.keys, idx)
        


# pool2d.cc
class Pool2DKey(KeyCompare):
    KEY_LENGTH = Tensor.MAX_KEY_LENGTH + 7
    def __init__(self, _input: Tensor, _type: OpType,
                 _kernelH: Tensor, _kernelW: Tensor, 
                 _strideH: int, _strideW:int, 
                 _padding: PaddingMode, _activation: ActiMode) -> None:
        super().__init__()
        self.keys = [0]*self.KEY_LENGTH
        idx = 0
        self.keys[: 7] = [_kernelH, _kernelW, _strideH, _strideW, _padding, _activation, _type]
        idx += 7
        _input.serialize(self.keys, idx)


class ReshapeKey(KeyCompare):
    KEY_LENGTH = Tensor.MAX_KEY_LENGTH + MAX_DIM + 1

    def __init__(self, _input: Tensor, shape: list[int]) -> None:
        self.keys = [0]*self.KEY_LENGTH
        idx = 0
        self.keys[idx] = len(shape)
        idx += 1
        for i in range(len(shape)):
            self.keys[idx] = shape[i]
            idx += 1
        _input.serialize(self.keys, idx)
        

class TransposeKey(KeyCompare):
    KEY_LENGTH = Tensor.MAX_KEY_LENGTH + 2

    def __init__(self, _input: Tensor, perm: list[int], _shuffle: bool) -> None:
        from utils import permutation_to_index
        self.keys = [0]*self.KEY_LENGTH
        idx = 0
        self.keys[idx] = permutation_to_index(perm)
        idx += 1
        self.keys[idx] = int(_shuffle)
        idx += 1
        _input.serialize(self.keys, idx)
      

class Model:
    def __init__(self) -> None:
        self.global_unique_id = 100
        self.concat = SortedDict(key=ConcatKey)
        self.conv2d = SortedDict(key=Conv2DKey)
        self.element = SortedDict(key=ElementKey)
        self.reshape = SortedDict(key=ReshapeKey)
        self.activation = SortedDict(key=ActivationKey)
        self.pool2d = SortedDict(key=Pool2DKey)
        self.transpose = SortedDict(key=TransposeKey)
        self.matmul = SortedDict(key=MatmulKey)

    def create_input(self, _input: Tensor, _type: OpType) -> Op:
        assert _type == OpType.OP_INPUT
        ret = Op()
        ret.ptr = NoOp(_input, _type)
        ret.guid = self.global_unique_id
        self.global_unique_id += 1
        return ret

    def create_weight(self,  _weight: Tensor, _type: OpType) -> Op:
        assert _type == OpType.OP_WEIGHT
        if _weight.data_ptr is None:
            logging.error("[MODEL] _weight_ptr is None")
            raise NotImplementedError
        ret = Op()
        ret.ptr = NoOp(_weight, _type)
        ret.guid = self.global_unique_id
        self.global_unique_id += 1
        return ret
    
    def get_or_create_element(self, type: OpType, t1: Tensor, t2: Tensor) -> Op:
        if not self.broadcastable(t1, t2):
            return Op.INVALID_OP
        key = ElementKey(t1, t2, type)
        if key in self.element:
            eleOp = self.element[key]
        else:
            eleOp = Element(type, t1, t2)
            # measure_element_cost(eleOp);
            self.element[key] = eleOp
        ret = Op()
        ret.ptr = eleOp
        ret.guid = self.global_unique_id
        self.global_unique_id += 1
        return ret
    
    def get_or_create_concat(self, axis: int, n: int, 
                             _inputs: List[Tensor], _needCopy: List[bool],
                             ) -> Op:
        for i in range(n):
            if _inputs[i].numDim != _inputs[0].numDim:
                return Op.INVALID_OP
            for j in range(_inputs[0].numDim):
                if (j != axis) and (_inputs[i].dim[j] != _inputs[0].dim[j]):
                    return Op.INVALID_OP
        key = ConcatKey(axis, n, _inputs, _needCopy)
        if key in self.concat:
            concatOp = self.concat[key]
        else:
            concatOp = Concat(axis, n, _inputs, _needCopy)
            # measure_concat_cost(concatOp)
            self.concat[key] = concatOp
        ret = Op()
        ret.ptr = concatOp
        ret.guid = self.global_unique_id
        self.global_unique_id += 1
        return ret
    
    def get_or_create_conv2d(self, _input: Tensor, _weight: Tensor,
                             _strideH: int, _strideW: int, 
                             _padding: PaddingMode, _activation: ActiMode,
                             ) -> Op:
        if _input.dim[1] % _weight.dim[1] != 0:
            return Op.INVALID_OP
        key = Conv2DKey(_input, _weight, _strideH, _strideW, _padding, _activation)
        
        if key in self.conv2d:
            convOp = self.conv2d[key]
        else:
            convOp = Conv2D(_input, _weight, _strideH, _strideW, _padding, _activation)
            # to implement
            # measure_conv2d_cost(convOp);
            self.conv2d[key] = convOp
        ret = Op()
        ret.ptr = convOp
        ret.guid = self.global_unique_id
        self.global_unique_id += 1
        return ret
    
    def get_or_create_pool2d(self, _input: Tensor, _weight: Tensor,
                             _type: OpType,
                             _kernelH: int, _kernelW: int,
                             _strideH: int, _strideW: int, 
                             _padding: PaddingMode, _activation: ActiMode,
                             ) -> Op:
        key = Pool2DKey(_input, _type, _kernelH, _kernelW, _strideH, _strideW, _padding, _activation)
        if key in self.pool2d:
            poolOp = self.pool2d[key]
        else:
            poolOp = Pool2D(_input, _weight, _type, _kernelH, _kernelW, _strideH, _strideW, _padding, _activation)
            # measure_pool2d_cost
            self.pool2d[key] = poolOp
        ret = Op()
        ret.ptr = poolOp
        ret.guid = self.global_unique_id
        self.global_unique_id += 1
        return ret
    
    def get_or_create_activation(self, _input: Tensor, _type: OpType, _inPlace: bool) -> Op:
        key = ActivationKey(_input, _type, _inPlace)
        if key in self.activation:
            actOp = self.activation[key]
        else:
            actOp = Activation(_input, _type, _inPlace)
            # measure_activation_cost
            self.activation[key] = actOp
        ret = Op()
        ret.ptr= actOp
        ret.guid = self.global_unique_id
        self.global_unique_id += 1
        return ret
    
    def get_or_create_matmul(self, _input: Tensor, _weight: Tensor, _acti: ActiMode) -> Op:
        if _input.numDim != _weight.numDim:
            return Op.INVALID_OP
        for i in range(_input.numDim - 2):
            if _input.dim[i] != _weight.dim[i]:
                return Op.INVALID_OP
        if _input.dim[_input.numDim-1] != _weight.dim[_weight.numDim-2]:
            return Op.INVALID_OP
        key = MatmulKey(_input, _weight, _acti)
        if key in self.matmul:
            matmulOp = self.matmul[key]
        else:
            matmulOp = Matmul(_input, _weight, _acti)
            # measure_matmul_cost(matmulOp)
            self.matmul[key] = matmulOp
        ret = Op()
        ret.guid = self.global_unique_id 
        self.global_unique_id += 1
        ret.ptr = matmulOp
        return ret
    
    def get_or_create_reshape(self, _input: Tensor, _shape: list[int]) -> Op:
        key = ReshapeKey(_input, _shape)
        if key in self.reshape:
            reshapeOp = self.reshape[key]
        else:
            reshapeOp = Reshape(_input, _shape)
            # measure_reshape_cost
            self.reshape[key] = reshapeOp
        ret = Op()
        ret.ptr= reshapeOp
        ret.guid = self.global_unique_id
        self.global_unique_id += 1
        return ret
    
    def get_or_create_transpose(self, _input: Tensor, perm: list[int], _shuffle: bool) -> Op:
        key = TransposeKey(_input, perm, _shuffle)
        if key in self.transpose:
            transposeOp = self.transpose[key]
        else:
            transposeOp = Transpose(_input, perm, _shuffle)
        ret = Op()
        ret.ptr= transposeOp
        ret.guid = self.global_unique_id
        self.global_unique_id += 1
        return ret
    
    # element.cc
    def broadcastable(self, t1: Tensor, t2: Tensor) -> bool:
        num_dim = min(t1.numDim, t2.numDim)
        for dim in range(num_dim):
            if ((t1.dim[t1.numDim-1-dim] != 1)
                and (t2.dim[t2.numDim-1-dim] != 1)
                and (t1.dim[t1.numDim-1-dim] != t2.dim[t2.numDim-1-dim])):
                return False
        return True