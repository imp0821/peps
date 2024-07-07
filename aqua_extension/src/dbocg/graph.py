from const import *
from tensor import TensorHandle, Tensor
from model import Model
from op import Op, Reshape
from substitution import GraphXfer

import copy
from typing import Optional, List, Dict
from sortedcontainers import SortedSet, SortedDict
import heapq
import logging

class Edge:
    def __init__(self, _srcOp: Optional[Op]= None, _dstOp: Optional[Op]= None,
                 _srcIdx: int= -1, _dstIdx: int= -1) -> None:
        if not _srcOp:
            self.srcOp = Op.INVALID_OP
            self.dstOp = Op.INVALID_OP
            self.srcIdx = -1
            self.dstIdx = -1
        else:
            self.srcOp = _srcOp
            self.dstOp = _dstOp
            self.srcIdx = _srcIdx
            self.dstIdx = _dstIdx
    
    def __eq__(self, other: 'Edge'):
        return (self.srcOp, self.dstOp, self.srcIdx, self.dstIdx) == (other.srcOp, other.dstOp, other.srcIdx, other.dstIdx)

    def __lt__(self, other: 'Edge'):
        if self.srcOp != other.srcOp:
            return self.srcOp < other.srcOp
        if self.dstOp != other.dstOp:
            return self.dstOp < other.dstOp
        if self.srcIdx != other.srcIdx:
            return self.srcIdx < other.srcIdx
        if self.dstIdx != other.dstIdx:
            return self.dstIdx < other.dstIdx
        return False
    
    def __hash__(self):
        return hash((self.srcOp, self.dstOp, self.srcIdx, self.dstIdx))
    

class OpMap:
    def __init__(self) -> None:
        self._map = SortedDict()

    def __getitem__(self, op: Op):
        if op not in self._map:
            self._map[op] = SortedSet(key=Edge)
        return self._map[op]

    def __setitem__(self, op, edge_set):
        self._map[op] = edge_set

    def __contains__(self, op: Op):
        return op in self._map

    # iterator interface
    def __iter__(self):
        return iter(self._map)

    # kv iterator interface
    def items(self):
        return self._map.items()
    
    def __repr__(self):
        return repr(self._map)
    
    def size(self):
        return len(self._map)
    
    def remove(self, op: Op):
        _ = self._map.pop(op, None)
    
    

class Graph:
    subst_history: List['GraphSubst']
    
    class GraphSubst:
        srcOps: List[Op]
        dstOps: List[Op]
        def __init__(self) -> None:
            self.srcOps = list()
            self.dstOps = list()
            
    def __init__(self) -> None:
        self.model = Model()
        self.totalCost = 0.0
        self.inEdges = OpMap()
        self.outEdges = OpMap()
        self.subst_history = list()
        # std::map<Op, std::set<Edge, EdgeCompare>, OpCompare> inEdges, outEdges; 

    def __lt__(self, other: 'Graph'):
        return self.totalCost < other.totalCost

    def new_input(self, dims: list[int]) -> Tensor: 
        assert len(dims) < 16
       
        _input = Tensor(len(dims), dims, GUID_INPUT)
        # Always create new operator for input
        op = self.model.create_input(_input, OpType.OP_INPUT)
        self.add_edge(_input.op, op, _input.idx, 0)
        # print(f"new_input: srcOp({_input.op.guid}), dstOp({op.guid})")
        t = copy.deepcopy(op.ptr.outputs[0])
        t.op = op
        return t

    def total_cost(self) -> float:
        if self.totalCost > 0:
            return self.totalCost
        total = float(0)
        for inEdgeOp in self.inEdges:
            assert isinstance(inEdgeOp, Op)
            total += inEdgeOp.ptr.opCost
        self.totalCost = total
        return total

    def new_weight(self, dims: list[int], data) -> Tensor:
        assert len(dims) < 16
        _weight = Tensor(len(dims), dims, GUID_WEIGHT, data)
        # Always create new operator for input
        op = self.model.create_weight(_weight, OpType.OP_WEIGHT)
        # print(f"new_weight: srcOp({_weight.op.guid}), dstOp({op.guid})")
        self.add_edge(_weight.op, op, _weight.idx, 0)
        t = copy.deepcopy(op.ptr.outputs[0])
        t.op = op
        return t

    def add_edge(self, srcOp: Op, dstOp: Op, srcIdx: int, dstIdx: int) -> None:
        assert dstOp.guid != OpType.OP_WEIGHT
        if dstOp not in self.inEdges:
            self.inEdges[dstOp]  
        if srcOp not in self.outEdges:
            self.outEdges[srcOp]  

        edge = Edge(srcOp, dstOp, srcIdx, dstIdx)
        self.inEdges[dstOp].add(edge)
        self.outEdges[srcOp].add(edge)
    
    def has_edge(self, srcOp: Op, dstOp: Op, srcIdx: int, dstIdx: int) -> bool:
        e = Edge(srcOp, dstOp, srcIdx, dstIdx)
        return e in self.inEdges[dstOp]
    
    def remove_edge(self, e: Edge) -> None:
        assert e.srcOp in self.outEdges
        assert e.dstOp in self.inEdges
        self.outEdges[e.srcOp].remove(e)
        self.inEdges[e.dstOp].remove(e)
    
    def remove_node(self, oldOp: Op) -> None:
        assert oldOp in self.outEdges
        assert self.num_out_edges(oldOp) == 0
        inSet: SortedSet = self.inEdges[oldOp]
        inList: List[Edge] = list()
        for e in inSet:
            inList.append(e)
        for e in inList:
            self.remove_edge(e)
        assert self.num_in_edges(oldOp) == 0
        self.inEdges.remove(oldOp)
        self.outEdges.remove(oldOp)
    
    def conv2d(self, _input: Tensor, _weight: Tensor, _strideH: int, _strideW: int,
                    _padding: PaddingMode, _activation: ActiMode) -> Tensor:
        op = self.model.get_or_create_conv2d(_input, _weight, 
                                             _strideH, _strideW, _padding, _activation)
        assert op != Op.INVALID_OP
        self.add_edge(_input.op, op, _input.idx, 0)
        self.add_edge(_weight.op, op, _weight.idx, 1)
        t = copy.deepcopy(op.ptr.outputs[0])
        t.op = op
        return t
    
    def element(self, type: OpType, t1: Tensor, t2: Tensor) -> Tensor:
        if not self.model.broadcastable(t1, t2):
            logging.error("inputs could not be broadcast together")
            assert False
        op = self.model.get_or_create_element(type, t1, t2)
        self.add_edge(t1.op, op, t1.idx, 0)
        self.add_edge(t2.op, op, t2.idx, 1)
        t = copy.deepcopy(op.ptr.outputs[0])
        t.op = op
        return t

    def matmul(self, _input: Tensor, _weight: Tensor, 
               acti: ActiMode = ActiMode.AC_MODE_NONE,
               ) -> Tensor:
        op = self.model.get_or_create_matmul(_input, _weight, acti)
        assert op != Op.INVALID_OP
        self.add_edge(_input.op, op, _input.idx, 0)
        self.add_edge(_weight.op, op, _weight.idx, 1)
        t = copy.deepcopy(op.ptr.outputs[0])
        t.op = op
        return t
    
    def pool2d_avg(self, _input: Tensor, 
                   _kernelH: int, _kernelW: int, 
                   _strideH: int, _strideW: int,
                   _padding: PaddingMode, _activation: ActiMode,
                    ) -> Tensor:
        num = _input.dim[1] * _kernelH * _kernelW
        data = [1.0 / (_kernelH * _kernelW) for _ in range(num)]
        dims = [_input.dim[1], 1, _kernelH, _kernelW]
        weight = self.new_weight(dims, data)
        op = self.model.get_or_create_pool2d(_input, weight, OpType.OP_POOL2D_AVG, _kernelH, _kernelW, _strideH, _strideW, _padding, _activation)
        self.add_edge(_input.op, op, _input.idx, 0)
        self.add_edge(weight.op, op, weight.idx, 1)
        t = copy.deepcopy(op.ptr.outputs[0])
        t.op = op
        return t
    
    def relu(self, _input: Tensor, _inPlace: Optional[bool] = False) -> Tensor:
        op = self.model.get_or_create_activation(_input, OpType.OP_RELU, _inPlace)
        assert op != Op.INVALID_OP
        self.add_edge(_input.op, op, _input.idx, 0)
        t = copy.deepcopy(op.ptr.outputs[0])
        t.op = op
        return t
    
    def reshape(self, _input: Tensor, _shape: list[int]) -> Tensor:
        myshape = copy.deepcopy(_shape)
        # replace zeros with input dims
        for i in range(len(myshape)):
            if myshape[i] == 0:
                myshape[i] = _input.dim[i]
        input_size = _input.volume()
        # # replace -1 with actual size
        for i in range(len(myshape)):
            if myshape[i] != -1:
                assert input_size % myshape[i] == 0
                input_size = input_size // myshape[i]
        for i in range(len(myshape)):
            if myshape[i] == -1:
                myshape[i] = input_size
                input_size = 1
        assert input_size == 1
        op = self.model.get_or_create_reshape(_input, myshape)
        self.add_edge(_input.op, op, _input.idx, 0)
        t = copy.deepcopy(op.ptr.outputs[0])
        t.op = op
        return t
    
    def transpose(self, _input: Tensor, 
                  perm: list[int], 
                  _shuffle: bool = False) -> Tensor:
        op = self.model.get_or_create_transpose(_input, perm, _shuffle)
        self.add_edge(_input.op, op, _input.idx, 0)
        t = copy.deepcopy(op.ptr.outputs[0])
        t.op = op
        return t
    
    
    # helper function
    def num_in_edges(self, op: Op) -> int:
        return len(self.inEdges[op])
    
    def num_out_edges(self, op: Op) -> int:
        return len(self.outEdges[op])
    
    def hash(self) -> int:
        total = 0
        for op, edge_set in self.inEdges.items():
            my = 17 * 31 + id(op.ptr)
            for edge in edge_set:
                my = my * 31 + hash(edge.srcOp.ptr)
                my = my * 31 + hash(edge.srcIdx)
                my = my * 31 + hash(edge.dstIdx)
                my &= 0xFFFFFFFFFFFFFFFF  # ensure my not exceed size_t size
            total += my
            total &= 0xFFFFFFFFFFFFFFFF
        return total
    
    def check_correctness(self) -> bool:
        okay = True
        for op, outList in self.outEdges.items():
            # print(op)
            for e in outList:
                if not self.has_edge(e.srcOp, e.dstOp, e.srcIdx, e.dstIdx):
                    assert False, "has edge"
                if e.srcOp.ptr is None:
                    continue
                srcTensor = e.srcOp.ptr.outputs[e.srcIdx]
                dstTensor = e.dstOp.ptr.inputs[e.dstIdx]
                # print(srcTensor.dim, e.srcOp)
                # print(dstTensor.dim, e.dstOp)
                if srcTensor.numDim != dstTensor.numDim:
                    assert False
                for i in range(srcTensor.numDim):
                    if srcTensor.dim[i] != dstTensor.dim[i]:
                        if e.srcOp.ptr.type == OpType.OP_CONV2D and i == 0:
                            continue
                        if self.model.broadcastable(srcTensor, dstTensor) and e.dstOp.ptr.type == OpType.OP_EW_ADD:
                            continue
                        assert False
                    if srcTensor.stride[i] != dstTensor.stride[i]:
                        pass  # to do
        return okay
    
    def has_loop(self) -> bool:
        todos: Dict[Op, int] = {}
        opList: List[Op] = []
        for op, inList in self.inEdges.items():
            cnt = 0
            for e1 in inList:
                if e1.srcOp.guid > GUID_PRESERVED:
                    cnt += 1
            todos[op] = cnt
            if todos[op] == 0:
                opList.append(op)
        i = 0
        while i < len(opList):
            op = opList[i]
            i += 1
            outList = self.outEdges[op]
            for e2 in outList:
                todos[e2.dstOp] -= 1
                if todos[e2.dstOp] == 0:
                    opList.append(e2.dstOp)
        return len(opList) < self.inEdges.size()
        
    
    def optimize(self, alpha: float, budget: float, print_subst: bool) -> 'Graph':
        xfers: list[GraphXfer] = list()
        # for i in range(1, 3, 1):
        #     for j in range(0, 2, 1):
        #         pad_mode = PaddingMode.PD_MODE_SAME if j == 0 else PaddingMode.PD_MODE_VALID
        #         xfers.append(GraphXfer.create_conv_relu(self.model, i, i, pad_mode))
        # xfers.append(GraphXfer.create_conv_relu(self.model, 1, 1, PaddingMode.PD_MODE_SAME))
        
        graph_subst_file = "/home/pyc/workspace/TASO/graph_subst.pb"
        GraphXfer.load_graph_xfer_from_pb_file(self.model, xfers, graph_subst_file)
        xfers = xfers[131:]
        candidates = []
        hashmap = set()
        heapq.heappush(candidates, self)
        hashmap.add(self.hash())
        bestGraph: Graph = self
        bestCost = self.total_cost()
        
        counter = 0
        maxNumOps = self.inEdges.size()
        
        while candidates:
            subGraph: Graph = heapq.heappop(candidates)
            if subGraph.totalCost < bestCost:
                del bestGraph
                bestCost = subGraph.total_cost()
                bestGraph = subGraph
            if counter > budget:
                break
            if counter % 1 == 0:
                pass
            counter += 1
            # for op, edge_list in self.inEdges.items():
            #     print(op)
                 
            for i in range(len(xfers)):
                print(f"run {i}")
                xfers[i].run(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps)
        bestGraph = bestGraph.preprocess_weights()
        return bestGraph
            
    def preprocess_weights(self):
        newGraph = Graph()
        newGraph.subst_history = self.subst_history
        # Step 1: clone the input graph
        for k, inList in self.inEdges.items():
            for e in inList:
                newGraph.add_edge(e.srcOp, e.dstOp, e.srcIdx, e.dstIdx)
        # Step 2: iteratively process the weights
        while True:
            change = False
            for op, inList in newGraph.inEdges.items():
                if op.ptr.type in [OpType.OP_INPUT, OpType.OP_WEIGHT]:
                    continue
                elif op.ptr.type == OpType.OP_TRANSPOSE:
                    continue
                allWeights = True
                for e in inList:
                    if e.srcOp.ptr.type != OpType.OP_WEIGHT:
                        allWeights = False
                        break
                if allWeights:
                    pass
                    # print("allWeights")
                    # for e in inList:
                    #     assert e.srcOp.ptr.outputs[e.srcIdx].data_ptr is not None
                    #     assert op.ptr.inputs[e.dstIdx].has_same_shape_stride_split(
                    #         e.srcOp.ptr.outputs[e.srcIdx])
                    #     op.ptr.inputs[e.dstIdx].data_ptr = \
                    #         e.srcOp.ptr.outputs[e.srcIdx].data_ptr
                    # op.ptr.map()
                    # op.ptr.forward(True)  # block
                    # tensor = newGraph.new_weight(op.ptr.outputs[0])
                    # newGraph.replace_node(op, tensor.op)
                    # op.ptr.unmap()
                    # newGraph.remove_node(op)
                    # change = True
                    # break
            # Stop if we didn't make any change
            if not change:
                break
        # Remove isolated nodes    
        todos: Dict[Op, int] = {}
        weightList: List[Op] = []
        weightOps = set()
        for op, inList in newGraph.inEdges.items():
            cnt = 0
            for e in inList:
                if e.srcOp.guid != GUID_WEIGHT:
                    cnt += 1
            todos[op] = cnt
            if cnt == 0:
                weightList.append(op)
        i = 0
        while i < len(weightList):
            op = weightList[i]
            i += 1
            weightOps.add(op)
            outList = newGraph.outEdges[op]
            for e in outList:
                todos[e.dstOp] -= 1
                if todos[e.dstOp] == 0:
                    weightList.append(e.dstOp)
        while True:
            change = False
            for op, inList in newGraph.inEdges.items():
                if op in weightOps and newGraph.num_out_edges(op) == 0:
                    newGraph.remove_node(op)
                    change = True
                    break
            if not change:
                break
        return newGraph      
    
    def get_input_edges(self, op: Op):
        assert op in self.inEdges
        inList = self.inEdges[op]
        inEdges: List[Edge] = [None]* len(inList)
        for e in inList:
            inEdges[e.dstIdx] = e
        if op.ptr.type == OpType.OP_POOL2D_MAX or op.ptr.type == OpType.OP_POOL2D_AVG:
            assert len(inEdges) == 1 or len(inEdges) == 2
            # manually delete the second input for pool2d
            if len(inEdges) == 2: 
                inEdges.pop()
        return inEdges
    
    def get_input_dims(self, op: Op, idx: int):
        assert op.ptr.numInputs > idx
        ndim = op.ptr.inputs[idx].numDim
        dims = [None] * ndim
        for i in range(ndim):
            dims[i] = op.ptr.inputs[idx].dim[i]
        return dims

    def get_output_dims(self, op: Op, idx: int):
        assert op.ptr.numOutputs > idx
        ndim = op.ptr.outputs[idx].numDim
        dims = [None] * ndim
        for i in range(ndim):
            dims[i] = op.ptr.outputs[idx].dim[i]
        return dims
    
    def get_weight_value(self, op: Op):
        assert op.ptr.type == OpType.OP_WEIGHT
        assert op.ptr.numInputs == 1 
        assert op.ptr.numOutputs == 1
        assert op.ptr.inputs[0].data_ptr is not None 
        # import numpy 
        # assert type(op.ptr.inputs[0].data_ptr) == numpy.ndarray
        return op.ptr.inputs[0].data_ptr
        

    def get_operator_list(self):
        ops: List[Op] = list()
        todos = {}
        opList: List[Op] = list()
        
        for op, inList in self.inEdges.items():
            cnt = sum(1 for inEdge in inList if inEdge.srcOp.guid > GUID_PRESERVED)
            
            todos[op] = cnt
            if cnt == 0:
                opList.append(op)
        
        cnt, i = 0, 0
        while i < len(opList):
            op = opList[i]
            i += 1
            # print(op, op.ptr.type)
            if not(op.ptr.type == OpType.OP_INPUT or op.ptr.type == OpType.OP_WEIGHT):
                ops.append(op)
                cnt += 1

            for outEdge in self.outEdges[op]:
                todos[outEdge.dstOp] -= 1
                if todos[outEdge.dstOp] == 0:
                    opList.append(outEdge.dstOp)
        
        assert len(opList) == self.inEdges.size()
        return ops
    
    def get_operator_type(self, op: Op):
        from const import op_table
        origin_type = op.ptr.type
        if origin_type in op_table:
            return op_table[origin_type]
        else: 
            assert False, 'Undefined type: {}'.format(type)
    
    def get_operator_attr(self, op: Op, attrname):
        if attrname == 'kernel_shape':
            kh, kw = [0], [0]
            assert op.ptr.get_int_parameter(PMParameter.PM_KERNEL_H, kh)
            assert op.ptr.get_int_parameter(PMParameter.PM_KERNEL_H, kw)
            return [kh[0], kw[0]]
        elif attrname == 'strides':
            sh, sw = [0], [0]
            assert op.ptr.get_int_parameter(PMParameter.PM_STRIDE_H, sh)
            assert op.ptr.get_int_parameter(PMParameter.PM_STRIDE_W, sw)
            return [sh[0], sw[0]]
        elif attrname == 'pads':
            pm = [None]
            assert op.ptr.get_int_parameter(PMParameter.PM_PAD, pm)
            pm = pm[0]
            if pm == PaddingMode.PD_MODE_VALID:
                return [0, 0, 0, 0]
            assert pm == PaddingMode.PD_MODE_SAME
            dims = self.get_input_dims(op, 0)
            # print(dims)
            assert len(dims) == 4, "input tensor must be 4 dim for pads attribute"
            kh, kw, sh, sw = [0], [0], [0], [0]
            assert op.ptr.get_int_parameter(PMParameter.PM_KERNEL_H, kh)
            assert op.ptr.get_int_parameter(PMParameter.PM_KERNEL_H, kw)
            assert op.ptr.get_int_parameter(PMParameter.PM_STRIDE_H, sh)
            assert op.ptr.get_int_parameter(PMParameter.PM_STRIDE_W, sw)
            kh, kw, sh, sw = kh[0], kw[0], sh[0], sw[0]
            inputH = dims[2]
            inputW = dims[3]
            if inputH % sh == 0:
                padH = max(kh - sh, 0)
            else:
                padH = max(kh - (inputH % sh), 0)
            if inputW % sw == 0:
                padW = max(kw - sw, 0)
            else:
                padW = max(kw - (inputW % sw), 0)
            # Ensure padding is same on both sides
            if padH % 2 == 1:
                padH += 1
            if padW % 2 == 1:
                padW += 1
            return [padH // 2, padW // 2, padH - padH // 2, padW - padW // 2]