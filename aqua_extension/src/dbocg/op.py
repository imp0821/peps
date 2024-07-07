from typing import Optional
from const import *
from const import OpType
from tensor import Tensor, SplitInfo
from abc import ABC, abstractmethod
from typing import List


# modify: remove model in OpBase
class OpBase(ABC):
    type: OpType
    opCost: float 
    inputs: list[Tensor]
    outputs: list[Tensor]

    def __init__(self, *args, type: OpType) -> None:
        self.numInputs = len(args)
        assert(self.numInputs <= MAX_NUM_INPUTS)
        if self.numInputs == 0:
            assert(type == OpType.OP_CONSTANT_POOL)

        # self.model = model
        self.type = type
        self.opCost = 0.1

        self.inputs = list(args) + [Tensor()] * (MAX_NUM_INPUTS - len(args))
        self.outputs = [Tensor() for _ in range(MAX_NUM_OUTPUTS)]
        self.numOutputs = 0
    
    # @abstractmethod
    # def get_input_parameter(self, TNParameter, DIMParameter, int_):
    #     pass
    
    def get_int_parameter(self, para: PMParameter, value: list[int]) -> bool:
        if para == PMParameter.PM_OP_TYPE:
            value[0] = self.type
            return True
        elif para == PMParameter.PM_NUM_INPUTS:
            value[0] = self.numInputs
            return True
        elif para == PMParameter.PM_NUM_OUTPUTS:
            value[0] = self.numOutputs
            return True
        else:
            return False
    
    def get_input_parameter(self, 
                            tnp: TNParameter, 
                            dim: DIMParameter, 
                            value: list[int]) -> bool:
        inputIdx, dimIdx = 0, 0
        if tnp in [TNParameter.IN_5, TNParameter.IN_4, TNParameter.IN_3, TNParameter.IN_2, TNParameter.IN_1]:
            inputIdx += 1
        elif tnp == TNParameter.IN_0: 
            pass
        else:
            return False
        if inputIdx >= self.numInputs:
            return False
        
        if dim in [DIMParameter.DIM_3, DIMParameter.DIM_2, DIMParameter.DIM_1]:
            dimIdx += 1
        elif dim == DIMParameter.DIM_0:
            pass 
        elif dim == DIMParameter.DIM_ND:
            value[0] = self.inputs[inputIdx].numDim
            return True
        else:
            return False

        if dimIdx >= self.inputs[inputIdx].numDim:
            return False
        value[0] = self.inputs[inputIdx].dim[dimIdx]
        return True
    
    # @abstractmethod
    # def get_float_parameter(self, PMParameter, float_):
    #     pass
    
    # @abstractmethod
    # def forward(self, block=False):
    #     pass
    
    # @abstractmethod
    # def map(self):
    #     pass
    
    # @abstractmethod
    # def unmap(self):
    #     pass
    
    # @abstractmethod
    # def collect_costs(self, exe_time, flops, mem_acc, num_kernels):
    #     pass
        
class NoOp(OpBase):
    def __init__(self, _input: Tensor, type: OpType) -> None:
        super().__init__(_input, type=type) 
        self.numOutputs = 1
        self.outputs[0] = _input
        self.outputs[0].idx = 0



class Activation(OpBase):
    def __init__(self, _input: Tensor, _type: OpType,
                 _inPlace: bool) -> None:
        super().__init__(_input, type=_type)
        self.inPlace = _inPlace
        self.numOutputs = 1
        self.outputs[0] = _input
        self.outputs[0].idx = 0


class Concat(OpBase):
    def __init__(self, _axis: int, n: int, 
                 _inputs: List[Tensor], _needCopy: List[bool],
                 ) -> None:
        super().__init__(*_inputs, type=OpType.OP_CONCAT)
        self.axis = _axis
        self.needCopy = [False] * MAX_NUM_INPUTS
        for i in range(n):
            self.needCopy[i] = _needCopy[i]
        self.numOutputs = 1
        self.outputs[0].numDim = _inputs[0].numDim
        for i in range(self.outputs[0].numDim):
            self.outputs[0].dim[i] = _inputs[0].dim[i]
        for i in range(self.outputs[0].numDim):
            if i != _axis:
                self.outputs[0].split[i] = _inputs[0].split[i]
                for j in range(1, n, 1):
                    self.outputs[0].split[i].combine(_inputs[j].split[i])
        self.outputs[0].split[_axis] = _inputs[0].split[_axis]
        for i in range(1, n, 1):
            self.outputs[0].split[_axis].merge(self.outputs[0].dim[_axis], _inputs[i].split[_axis])
            self.outputs[0].dim[_axis] += _inputs[i].dim[_axis]
        for i in range(self.outputs[0].numDim - 1, -1, -1):
            if i == self.outputs[0].numDim - 1:
                self.outputs[0].stride[i] = 1
            else: 
                self.outputs[0].stride[i] = self.outputs[0].stride[i+1] * self.outputs[0].dim[i+1]
        self.outputs[0].idx = 0
        

class Conv2D(OpBase):
    def __init__(self, _input: Tensor, _weight: Tensor,
                 _strideH: int, _strideW: int,
                 _padding: PaddingMode,
                 _activation: ActiMode) -> None:
        super().__init__(_input, _weight, type=OpType.OP_CONV2D)
        self.strideH = _strideH
        self.strideW = _strideW
        self.padding = _padding
        self.activation = _activation

        assert _input.numDim == 4
        assert _weight.numDim == 4
        assert _input.dim[1] % _weight.dim[1] == 0
        groups = _input.dim[1] / _weight.dim[1]
        assert _weight.dim[0] % groups == 0
        inputH = _input.dim[2]
        inputW = _input.dim[3]
        kernelH = _weight.dim[2]
        kernelW = _weight.dim[3]
        
        outputH, outputW = None, None
        if self.padding == PaddingMode.PD_MODE_SAME:
            outputH = (inputH + self.strideH - 1) // self.strideH
            outputW = (inputW + self.strideW - 1) // self.strideW
        elif self.padding == PaddingMode.PD_MODE_VALID:
            outputH = (inputH - kernelH) // self.strideH + 1
            outputW = (inputW - kernelW) // self.strideW + 1
        else:
            assert False

        self.numOutputs = 1
        self.outputs[0].numDim = 4
        self.outputs[0].dim[:4] = [_input.dim[0],_weight.dim[0],outputH,outputW]
        self.outputs[0].stride[3] = 1
        self.outputs[0].stride[2] = self.outputs[0].stride[3] * self.outputs[0].dim[3]
        self.outputs[0].stride[1] = self.outputs[0].stride[2] * self.outputs[0].dim[2]
        self.outputs[0].stride[0] = self.outputs[0].stride[1] * self.outputs[0].dim[1]
        
        # Set SplitInfo
        self.outputs[0].split[0] = _input.split[0]
        self.outputs[0].split[1] = _input.split[0]
        self.outputs[0].split[2] = _input.split[2]
        self.outputs[0].split[3] = _input.split[3]
         
        # Assume we cannot split the H and W dimension,
        # otherwise we need to extend Conv2DKey to include their SplitInfo
        assert self.outputs[0].split[2] == SplitInfo.NO_SPLIT
        assert self.outputs[0].split[3] == SplitInfo.NO_SPLIT
        self.outputs[0].idx = 0
    
    def get_int_parameter(self, para: PMParameter, value: list[int]) -> bool:
        if para == PMParameter.PM_GROUP:
            inputC = self.inputs[0].dim[1]
            weightC = self.inputs[1].dim[1]
            assert inputC % weightC == 0
            value[0] = inputC // weightC
            return True
        elif para == PMParameter.PM_KERNEL_H:
            value[0] = self.inputs[1].dim[2]
            return True
        elif para == PMParameter.PM_KERNEL_W:
            value[0] = self.inputs[1].dim[3]
            return True
        elif para == PMParameter.PM_STRIDE_H:
            value[0] = self.strideH
            return True
        elif para == PMParameter.PM_STRIDE_W:
            value[0] = self.strideW
            return True
        elif para == PMParameter.PM_PAD:
            value[0] = self.padding
            return True
        elif para == PMParameter.PM_ACTI:
            value[0] = self.activation
            return True
        else:
            return super().get_int_parameter(para, value)
        

class Pool2D(OpBase):
    def __init__(self, _input: Tensor, _weight: Tensor,
                 _type: OpType,
                 _kernelH: int, _kernelW: int,
                 _strideH: int, _strideW: int,
                 _padding: PaddingMode,
                 _activation: ActiMode) -> None:
        super().__init__(_input, _weight, type=_type)
        self.kernelH = _kernelH
        self.kernelW = _kernelW
        self.strideH = _strideH
        self.strideW = _strideW
        self.padding = _padding
        self.activation = _activation
        assert _type in [OpType.OP_POOL2D_AVG, OpType.OP_POOL2D_MAX]
        assert _input.numDim == 4
        inputH = _input.dim[2]
        inputW = _input.dim[3]
        
        outputH, outputW = None, None
        if self.padding == PaddingMode.PD_MODE_SAME:
            outputH = (inputH + self.strideH - 1) // self.strideH
            outputW = (inputW + self.strideW - 1) // self.strideW
        elif self.padding == PaddingMode.PD_MODE_VALID:
            outputH = (inputH - self.kernelH) // self.strideH + 1
            outputW = (inputW - self.kernelW) // self.strideW + 1
        else:
            assert False
     
        self.numOutputs = 1
        self.outputs[0].numDim = 4
        self.outputs[0].dim[0] = _input.dim[0]
        self.outputs[0].dim[1] = _input.dim[1]
        self.outputs[0].dim[2] = outputH
        self.outputs[0].dim[3] = outputW
        # Set strides
        self.outputs[0].stride[3] = 1
        self.outputs[0].stride[2] = self.outputs[0].dim[3] * self.outputs[0].stride[3]
        self.outputs[0].stride[1] = self.outputs[0].dim[2] * self.outputs[0].stride[2]
        self.outputs[0].stride[0] = self.outputs[0].dim[1] * self.outputs[0].stride[1]
        # Set SplitInfo
        self.outputs[0].split[0] = _input.split[0]
        self.outputs[0].split[1] = _input.split[1]
        self.outputs[0].split[2] = SplitInfo.NO_SPLIT
        self.outputs[0].split[3] = SplitInfo.NO_SPLIT
        self.outputs[0].idx = 0
    
    def get_int_parameter(self, para: PMParameter, value: list[int]) -> bool:
        if para == PMParameter.PM_KERNEL_H:
            value[0] = self.kernelH
            return True
        elif para == PMParameter.PM_KERNEL_W:
            value[0] = self.kernelW
            return True
        elif para == PMParameter.PM_STRIDE_H:
            value[0] = self.strideH
            return True
        elif para == PMParameter.PM_STRIDE_W:
            value[0] = self.strideW
            return True
        elif para == PMParameter.PM_PAD:
            value[0] = self.padding
            return True
        elif para == PMParameter.PM_ACTI:
            value[0] = self.activation
            return True
        else:
            return super().get_int_parameter(para, value)


class Element(OpBase):
    def __init__(self, type: OpType, _t1: Tensor, _t2: Tensor) -> None:
        super().__init__(_t1, _t2, type=type) 
        self.numOutputs = 1
        num_dim = max(_t1.numDim, _t2.numDim)
        self.outputs[0].numDim = num_dim
        total = 1
        for i in range(num_dim):
            t1_idx = _t1.numDim - 1 - i
            t2_idx = _t2.numDim - 1 - i
            out_idx = num_dim - 1 - i
            dim1 = 1 if t1_idx < 0 else _t1.dim[t1_idx]
            dim2 = 1 if t2_idx < 0 else _t2.dim[t2_idx]
            self.outputs[0].dim[out_idx] = max(dim1, dim2)
            self.outputs[0].stride[out_idx] = total
            total *= self.outputs[0].dim[out_idx]
            self.outputs[0].split[out_idx] = SplitInfo.NO_SPLIT
            if t1_idx >= 0 and _t1.dim[t1_idx] > 1:
                self.outputs[0].split[out_idx] = _t1.split[t1_idx]
                if t2_idx >= 0 and _t2.dim[t2_idx] > 1:
                    self.outputs[0].split[out_idx].combine(_t2.split[t2_idx])
            elif t2_idx >= 0 and _t2.dim[t2_idx] > 1:
                self.outputs[0].split[out_idx] = _t2.split[t2_idx]
        self.outputs[0].idx = 0
        
        
class Matmul(OpBase):
    def __init__(self, _input: Tensor, _weight: Tensor,
                 _activation: ActiMode) -> None:        
        super().__init__(_input, _weight, type=OpType.OP_MATMUL)
        self.activation = _activation
        numDim = _input.numDim
        outputs = self.outputs
        assert numDim == _weight.numDim
        for i in range(numDim - 2):
            assert _input.dim[i] == _weight.dim[i]
        assert _input.dim[numDim-1] == _weight.dim[numDim-2]
        self.numOutputs = 1
        # set dims and strides
        outputs[0].numDim = numDim
        for i in range(numDim - 1):
            outputs[0].dim[i] = _input.dim[i]
        outputs[0].dim[numDim-1] = _weight.dim[numDim-1]
    
        # self.set_layout(): MKL uses row-major.
        numDim = outputs[0].numDim
        outputs[0].stride[numDim-1] = 1
        outputs[0].stride[numDim-2] = outputs[0].dim[numDim-1]
        size = outputs[0].dim[numDim-2] * outputs[0].dim[numDim-1]
        for i in range(numDim - 3, -1, -1):
            outputs[0].stride[i] = size
            size *= outputs[0].dim[i]
        assert(size == outputs[0].volume());
        
        # set SplitInfo
        for i in range(numDim - 2):
            if _input.split[i] == _weight.split[i]:
                self.outputs[0].split[i] = _input.split[i]
            else:
                self.outputs[0].split[i] = SplitInfo.NO_SPLIT
        self.outputs[0].split[numDim-2] = _input.split[numDim-2]
        self.outputs[0].split[numDim-1] = _weight.split[numDim-1]
        self.outputs[0].idx = 0
    
    def get_int_parameter(self, para: PMParameter, value: list[int]) -> bool:
        if para == PMParameter.PM_ACTI:
            value[0] = self.activation
            return True
        else:
            return super().get_int_parameter(para, value)
        
        
class Reshape(OpBase):
    def __init__(self, _input: Tensor, _shape: list[int]) -> None:
        super().__init__(_input, type=OpType.OP_RESHAPE) 
        size = 1
        # set dims and strides
        self.numOutputs = 1
        self.outputs[0].numDim = len(_shape)
        for i in range(len(_shape)-1, -1, -1):
            self.outputs[0].dim[i] = _shape[i]
            self.outputs[0].stride[i] = size
            size *= _shape[i]
            self.outputs[0].split[i] = SplitInfo.NO_SPLIT
        assert _input.volume() == size
        self.outputs[0].idx = 0


class Transpose(OpBase):
    def __init__(self, _input: Tensor, _perm: list[int], _shuffle: bool) -> None:
        super().__init__(_input, type=OpType.OP_TRANSPOSE)
        self.shuffle = _shuffle
        assert self.shuffle
        
        from utils import permutation_to_index
        self.permIdx = permutation_to_index(_perm)
        assert _input.numDim == len(_perm)
        self.numOutputs = 1
        # set dims and strides
        self.outputs[0].numDim = _input.numDim
        for i in range(len(_perm)):
            self.outputs[0].dim[i] = _input.dim[_perm[i]]
            self.outputs[0].split[i] = _input.split[_perm[i]]
        if self.shuffle:
            size = 1
            for i in range(len(_perm) - 1, -1, -1):
                self.outputs[0].stride[i] = size
                size *= self.outputs[0].dim[i]
            assert size == self.outputs[0].volume()
        else:
            for i in range(len(_perm)):
                self.outputs[0].stride[i] = _input.stride[_perm[i]]
        self.outputs[0].idx = 0
        
    def get_int_parameter(self, para: PMParameter, value: list[int]) -> bool:
        if para == PMParameter.PM_NUMDIM:
            value[0] = self.outputs[0].numDim
            return True
        elif para == PMParameter.PM_PERM:
            value[0] = self.permIdx
            return True
        elif para == PMParameter.PM_OUTSHUFFLE:
            value[0] = self.shuffle
            return True
        else:
            return super().get_int_parameter(para, value) 
        

class Op:
    INVALID_OP = None  

    def __init__(self, 
                 _guid: Optional[int] = None, 
                 _ptr: Optional[OpBase] = None,
                 ) -> None:
        if _guid:
            self.guid = _guid 
            self.ptr = _ptr
        else:
            self.guid = GUID_INVALID
            self.ptr = None

    def __eq__(self, b: 'Op'):
        return self.guid == b.guid and self.ptr == b.ptr

    def __ne__(self, b: 'Op'):
        return not self.__eq__(b)

    def __lt__(self, b: 'Op'):
        if not isinstance(b, Op):
            return False
        
        if self.guid != b.guid:
            return self.guid < b.guid
        if self.ptr != b.ptr:
            return self.ptr < b.ptr
        return False
    
    def __hash__(self):
        return hash((self.guid, self.ptr))
    
    def copy_from(self, op: 'Op'):
        self.guid = op.guid
        self.ptr = op.ptr

    def op_to_string(self):
        return self.ptr.__class__.__name__
        # to implement
        # raise NotImplementedError

    def __str__(self):
        if self.ptr is not None:
            return self.op_to_string() + "_" + str(self.guid)
        else:
            return "UnmappedOp_" + str(self.guid)

Op.INVALID_OP = Op()