from const import OpType as OT
from const import *
from model import Model
from tensor import Tensor
from op import Op
import rules_pb2

from enum import Enum
from sortedcontainers import SortedDict
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
import google.protobuf as protobuf
from google.protobuf.text_format import Merge
import heapq


class Compare(Enum):
    COMPARE_EQ = 0
    COMPARE_NE = 1
    COMPARE_LT = 2
    COMPARE_LE = 3
    COMPARE_GT = 4
    COMPARE_GE = 5



class PMConstraint:
    comp: Compare
    para: PMParameter
    value: int

    def __init__(self, c: Compare, p: PMParameter, v: int) -> None:
        self.comp = c
        self.para = p 
        self.value = v
        

class TNConstraint:
    singlePara: bool
    comp: Compare
    para1: TNParameter
    para2: TNParameter
    dim1: DIMParameter
    dim2: DIMParameter
    value: int
    
    def __init__(self, c: Compare, 
                 p: PMParameter, d: DIMParameter,
                 p2: Optional[PMParameter] = None,
                 d2: Optional[DIMParameter] = None,
                 v: Optional[int] = 0) -> None:
        self.singlePara = False
        self.comp = c
        self.para1 = p 
        self.dim1 = d
        self.para2 = p2 if p2 else None 
        self.dim2 = d2 if d2 else None
        self.value = v if v else None


class OpX:
    type: OT
    mapOp: Op
    inputs: List['TensorX']
    outputs: List['TensorX']
    pmConstraints: List[PMConstraint]
    tnConstraints: List[TNConstraint]
    
    def __init__(self, _type: OT, *inputs, numOutputs: int=1) -> None:
        self.type = _type
        self.mapOp = Op()
        self.inputs: List[TensorX] = list(inputs)
        self.outputs: List[TensorX] = list()
        self.pmConstraints: List[PMConstraint] = list()
        self.tnConstraints: List[TNConstraint] = list()
       
        INPUT1_OPTYPE = [OT.OP_RESHAPE, OT.OP_TRANSPOSE, OT.OP_RELU, 
                         OT.OP_TANH, OT.OP_SIGMOID, OT.OP_MERGE_GCONV]
        INPUT2_OPTYPE = [OT.OP_CONV2D, OT.OP_EW_ADD, OT.OP_EW_MUL, 
                         OT.OP_POOL2D_AVG, OT.OP_CONCAT, OT.OP_MATMUL,
                         OT.OP_MUL, OT.OP_ENLARGE, OT.OP_BROADCAST_ADD]
        INPUT3_OPTYPE = [OT.OP_FUSE_CONV_BATCHNORM_ALPHA_VAR]
        INPUT4_OPTYPE = [OT.OP_FUSE_CONV_BATCHNORM_BIAS]
        INPUT5_OPTYPE = [OT.OP_BATCHNORM, OT.OP_FUSE_CONV_BATCHNORM]
        
        out = TensorX(self, 0)
        
        if self.type in (INPUT1_OPTYPE + INPUT2_OPTYPE + INPUT3_OPTYPE + INPUT4_OPTYPE + INPUT5_OPTYPE):
            self.outputs.append(out)
        elif self.type == OT.OP_SPLIT and numOutputs != 1:
            for i in range(numOutputs):
                self.outputs.append(TensorX(self, i))
        else:
            assert False

        assert len(self.outputs) != 0, "Invalid operation type"
    
    def add_pm_constraint(self, comp: Compare, para: PMParameter, value: int) -> bool:
        pmc = PMConstraint(comp, para, value)
        self.pmConstraints.append(pmc)
        return True
    
    def get_pm_constraint(self, para: PMParameter, value: List[int]) -> bool:
        for pmc in self.pmConstraints:
            if pmc.comp == Compare.COMPARE_EQ and pmc.para == para:
                value[0] = pmc.value
                return True
        return False
    
    def add_input_constraint(self, comp: Compare, 
                             para1: TNParameter, dim1: DIMParameter,
                             para2: Optional[TNParameter] = None, 
                             dim2: Optional[DIMParameter] = None,
                             value: Optional[int] = None,
                             ) -> bool:
        if para2 and dim2:
            tnc = TNConstraint(comp, para1, dim1, para2, dim2)
        elif value: 
            tnc = TNConstraint(comp, para1, dim1, value)  
        self.tnConstraints.append(tnc)
        return True


class TensorX:
    op: OpX
    idx: int
    
    def __init__(self,
                 _op: Optional[OpX] = None,
                 _idx: Optional[int] = None) -> None:
        self.op = _op
        self.idx = 0
        if _idx:
            self.idx = _idx
        # if _op and _idx:
        #     self.op, self.idx = _op, _idx
        # else:
        #     self.op, self.idx = None, 0
    
    def to_tensor(self, xfer: 'GraphXfer') -> Tensor:
        if self.op != None:
            assert self.op.mapOp.ptr != None 
            return self.op.mapOp.ptr.outputs[self.idx]
        else:
            it = xfer.mappedInputs.get(self.idx, None)
            assert it != None
            op = it[0][0]
            outIdx = it[0][1]
            return op.ptr.outputs[outIdx]

    def __lt__(self, other: 'TensorX'):
        if not isinstance(other, TensorX):
            return False
        if self.op != other.op:
            return id(self.op) < id(other.op)
        return self.idx < other.idx
    
    def __eq__(self, other: 'TensorX'):
        if not isinstance(other, TensorX):
            return False
        return self.op == other.op and self.idx == other.idx
    
    def __hash__(self) -> int:
        return hash((self.op, self.idx))


class GraphXfer:
    model: Model
    tensorId: int
    
    srcOps: List[OpX]
    dstOps: List[OpX]
    mappedOps: SortedDict(key=Op) # Op -> OpX
    mappedInputs: Dict[int, List[Tuple[Op, int]]]
    mappedOutputs: SortedDict(key=TensorX)
    
    def __init__(self, _model: Model) -> None:
        self.model = _model
        self.tensorId = 10
        self.srcOps = list()
        self.dstOps = list()
        self.mappedOps = SortedDict() #key = Op
        self.mappedInputs = defaultdict(list)
        self.mappedOutputs = SortedDict() # TensorX -> TensorX

    def new_tensor(self) -> TensorX:
        t = TensorX()
        t.op = None
        t.idx = self.tensorId
        self.tensorId += 1
        return t
    
    def check_tnConstraints(self, tnc: TNConstraint, op: Op) -> bool:
        actValueList = [0]
        expValueList = [0]
        if tnc.singlePara:
            assert op.ptr.get_input_parameter(tnc.para1, tnc.dim1, actValueList)
            expValueList = [tnc.value]
        else:
            assert op.ptr.get_input_parameter(tnc.para1, tnc.dim1, actValueList)
            assert op.ptr.get_input_parameter(tnc.para2, tnc.dim2, expValueList)
        actValue = actValueList[0]
        expValue = expValueList[0]
        tncSwitch = {
            Compare.COMPARE_EQ: lambda: actValue == expValue,
            Compare.COMPARE_NE: lambda: actValue != expValue,
            Compare.COMPARE_LT: lambda: actValue < expValue,
            Compare.COMPARE_LE: lambda: actValue <= expValue,
            Compare.COMPARE_GT: lambda: actValue > expValue,
            Compare.COMPARE_GE: lambda: actValue >= expValue,
        }
        tncResult = tncSwitch.get(tnc.comp, lambda: (_ for _ in ()).throw(AssertionError("Invalid key")))()
        return tncResult
    
    def can_match(self, srcOp: OpX, op: Op, graph):
        from graph import Graph
        assert isinstance(graph, Graph)
        if srcOp.type != op.ptr.type:
            return False
        # check num input tensors
        if len(srcOp.inputs) != op.ptr.numInputs:
            return False
        # check pmConstraints
        for pmc in srcOp.pmConstraints:
            actValueList = [0]
            assert op.ptr.get_int_parameter(pmc.para, actValueList)
            actValue = actValueList[0]
            pmcSwitch = {
                Compare.COMPARE_EQ: lambda: actValue == pmc.value,
                Compare.COMPARE_NE: lambda: actValue != pmc.value,
                Compare.COMPARE_LT: lambda: actValue < pmc.value,
                Compare.COMPARE_LE: lambda: actValue <= pmc.value,
                Compare.COMPARE_GT: lambda: actValue > pmc.value,
                Compare.COMPARE_GE: lambda: actValue >= pmc.value,
            }
            
            pmcResult = pmcSwitch.get(pmc.comp, lambda: (_ for _ in ()).throw(AssertionError("Invalid key")))()
            if not pmcResult:
                return False
        # check inputs
        # flag = False
        # if str(op) == "Element_177":
        #     for tt in self.mappedOps.keys():
        #         if str(tt) == "Element_169":
        #             flag = True
        
        
        newMapInputs: Dict[int, Tuple[Op, int]] = dict()
        for i in range(len(srcOp.inputs)):
            in_tensorX: TensorX = srcOp.inputs[i]
            in_idx = in_tensorX.idx
            
            if in_tensorX.op == None:
                # Dict[int, List[Tuple[Op, int]]]
                if in_idx in self.mappedInputs:
                    mappedOp = self.mappedInputs[in_idx][0][0]
                    mappedIdx = self.mappedInputs[in_idx][0][1]
                    if not graph.has_edge(mappedOp, op, mappedIdx, i):
                        return False
                else:
                    if in_idx in newMapInputs:
                        mappedOp = newMapInputs[in_idx][0]
                        mappedIdx = newMapInputs[in_idx][1]
                        if not graph.has_edge(mappedOp, op, mappedIdx, i):
                            return False
                    else:
                        edge_list = graph.inEdges[op]
                        for e in edge_list:
                            if e.dstIdx == i:
                                # print(e.srcOp.ptr, e.srcIdx)
                                newMapInputs[in_idx] = (e.srcOp, e.srcIdx)
            else:
                # intermediate tensor
                assert in_tensorX.op.mapOp.ptr is not None
                    # return e in self.inEdges[dstOp]
                if not graph.has_edge(in_tensorX.op.mapOp, op, in_tensorX.idx, i):
                    return False
       
        # check tnConstraints
        for tnc in srcOp.tnConstraints:
            tncResult = self.check_tnConstraints(tnc, op)
            if not tncResult:
                return False
        return True

    def match(self, srcOp: OpX, op: Op, graph) -> None:
        for i in range(len(srcOp.inputs)):
            in_TensorX: TensorX = srcOp.inputs[i]
            if in_TensorX.op == None:
                # update mappedInputs
                edge_list = graph.inEdges[op]
                for e in edge_list:
                    if e.dstIdx == i:
                        self.mappedInputs[in_TensorX.idx].append((e.srcOp, e.srcIdx))
        # Map srcOp to Op
        srcOp.mapOp = op
        self.mappedOps[op] = srcOp
    
    def unmatch(self, srcOp: OpX, op: Op, graph) -> None:
        for i in range(len(srcOp.inputs)):
            in_TensorX: TensorX = srcOp.inputs[i]
            if in_TensorX.op == None:
                # Update mappedInputsa
                del self.mappedInputs[in_TensorX.idx]
        # unmap op
        del self.mappedOps[op]
        srcOp.mapOp = Op()
    
    def create_activation(self, input: TensorX, 
                          type: OpType, 
                          isSrcOp: bool = True,
                          ) -> OpX:
        activation = OpX(type, input)
        return activation

    def create_concat(self, axis: int, numDim: int, 
                      n: int, ins: List[TensorX], 
                      isSrcOp: bool) -> OpX:
        concat = OpX(OT.OP_CONCAT, *ins)
        concat.add_pm_constraint(Compare.COMPARE_EQ, PMParameter.PM_AXIS, axis)
        concat.add_input_constraint(Compare.COMPARE_EQ, TNParameter.IN_0, DIMParameter.DIM_ND, value=numDim)
        for i in range(1, n):
            in_i = to_tn_parameter(True, i)
            concat.add_input_constraint(Compare.COMPARE_EQ, 
                                        TNParameter.IN_0, DIMParameter.DIM_ND,
                                        in_i, DIMParameter.DIM_ND)
            for j in range(numDim):
                dim_j = to_dim_parameter(j)
                if j != axis:
                    concat.add_input_constraint(Compare.COMPARE_EQ, 
                                                TNParameter.IN_0, dim_j,
                                                in_i, dim_j)
                 
        return concat
    
    def create_conv2d(self, input: TensorX, weight: TensorX,
                      strideH: int, strideW: int,
                      padding: PaddingMode,
                      activation: ActiMode,
                      isSrcOp: bool = True,
                      ) -> OpX:
        conv = OpX(OT.OP_CONV2D, input, weight)
        conv.add_pm_constraint(Compare.COMPARE_EQ, PMParameter.PM_STRIDE_H, strideH)
        conv.add_pm_constraint(Compare.COMPARE_EQ, PMParameter.PM_STRIDE_W, strideW)
        conv.add_pm_constraint(Compare.COMPARE_EQ, PMParameter.PM_PAD, padding)
        conv.add_pm_constraint(Compare.COMPARE_EQ, PMParameter.PM_ACTI, activation)
        return conv
    
    def create_element(self, input0: TensorX, input1: TensorX,
                       _type: OpType, isSrcOp: bool,
                       ) -> OpX:
        element = OpX(_type, input0, input1)
        return element 
    
    def create_matmul(self, input: TensorX, weight: TensorX, 
                          activation: ActiMode, 
                          isSrcOp: bool = True,
                          ) -> OpX:
        matmul = OpX(OpType.OP_MATMUL, input, weight)
        matmul.add_pm_constraint(Compare.COMPARE_EQ, PMParameter.PM_ACTI, activation)
        matmul.add_input_constraint(Compare.COMPARE_EQ, TNParameter.IN_0, DIMParameter.DIM_0, TNParameter.IN_1, DIMParameter.DIM_1)
        return matmul
    
    def create_split(self, input: TensorX, axis: int, n: int, isSrcOp: bool) -> OpX:
        split = OpX(OpType.OP_SPLIT, input, numOutputs=n)
        split.add_pm_constraint(Compare.COMPARE_EQ, PMParameter.PM_AXIS, axis)
        return split

    # def map_output(self, src: OpX, dst: OpX) -> bool:
    #     pass

    def map_output(self, src: TensorX, dst: TensorX) -> bool:
        self.mappedOutputs[src] = dst
        return True

    def get_parameter_from_pb(self, pbOp: rules_pb2.Operator,
                              pm: PMParameter, value: List[int]) -> bool:
        for i in range(len(pbOp.para)):
            if pbOp.para[i].key == pm.value:
                value[0] = pbOp.para[i].value
                return True
        return False
    
    # built-in substitutions
    @staticmethod
    def create_conv_relu(model: Model, 
                         strideH: int, strideW: int,
                         mode: PaddingMode) -> 'GraphXfer':
        subst = GraphXfer(model)
        input = subst.new_tensor()
        weight = subst.new_tensor()
        conv = subst.create_conv2d(input, weight, strideH, strideW, mode, ActiMode.AC_MODE_NONE)
        relu = subst.create_activation(conv.outputs[0], OT.OP_RELU)
        fuse = subst.create_conv2d(input, weight, strideH, strideW, mode, ActiMode.AC_MODE_RELU, False)
        subst.map_output(relu.outputs[0], fuse.outputs[0])
        subst.srcOps.append(conv)
        subst.srcOps.append(relu)
        subst.dstOps.append(fuse)
        return subst

    def create_operator_from_pb(self, pbOp: rules_pb2.Operator, 
                                mappedInputs: Dict[int, TensorX], 
                                isSrcOp: bool,
                                ) -> None:
        # Step 1: create inputs
        input_size = len(pbOp.input)
        # print(input_size)
        assert input_size < MAX_NUM_INPUTS
        inputs = [TensorX()] * input_size
        for i in range(input_size):
            tensor: rules_pb2.Tensor = pbOp.input[i]
            # print(tensor)
            if tensor.opId < 0:
                opId = tensor.opId
                if opId not in mappedInputs:
                    mappedInputs[opId] = self.new_tensor()   
                    assert isSrcOp 
                inputs[i] = mappedInputs[opId]
            else:
                opId = tensor.opId
                tsId = tensor.tsId
                if isSrcOp:
                    inputs[i] = self.srcOps[opId].outputs[tsId]
                else:
                    inputs[i] = self.dstOps[opId].outputs[tsId]
        # Step 2: create op
        op_type = OpType(pbOp.type)
        opx: OpX = None
        
        # print(op_type)
        if op_type == OpType.OP_CONV2D:
            assert input_size == 2
            strideH, strideW, padding, activation = [0], [0], [0], [0]
            assert self.get_parameter_from_pb(pbOp, PMParameter.PM_STRIDE_H, strideH)
            assert self.get_parameter_from_pb(pbOp, PMParameter.PM_STRIDE_W, strideW)
            assert self.get_parameter_from_pb(pbOp, PMParameter.PM_PAD, padding)
            assert self.get_parameter_from_pb(pbOp, PMParameter.PM_ACTI, activation)
            opx = self.create_conv2d(inputs[0], inputs[1], strideH[0], strideW[0], PaddingMode(padding[0]), ActiMode(activation[0]), isSrcOp)
            
        elif op_type == OpType.OP_CONCAT:
            numDim, axis = [0], [0]
            assert self.get_parameter_from_pb(pbOp, PMParameter.PM_AXIS, axis)
            assert self.get_parameter_from_pb(pbOp, PMParameter.PM_NUMDIM, numDim)
            opx = self.create_concat(axis[0], numDim[0], input_size, inputs, isSrcOp)
            
        elif op_type == OpType.OP_EW_ADD or op_type == OpType.OP_EW_MUL:
            assert input_size == 2
            opx = self.create_element(inputs[0], inputs[1], op_type, isSrcOp)
            
        elif op_type == OpType.OP_SPLIT:
            assert input_size == 1
            numOutputs, axis = [0], [0]
            assert self.get_parameter_from_pb(pbOp, PMParameter.PM_AXIS, axis)
            assert self.get_parameter_from_pb(pbOp, PMParameter.PM_NUM_OUTPUTS, numOutputs)
            opx = self.create_split(inputs[0], axis[0], numOutputs[0], isSrcOp)
            
        elif op_type in [OpType.OP_RELU, OpType.OP_SIGMOID, OpType.OP_TANH]:
            assert input_size == 1
            opx = self.create_activation(inputs[0], op_type)
            
        elif op_type == OpType.OP_MUL:
            pass
        elif op_type == OpType.OP_ENLARGE:
            pass
        elif op_type == OpType.OP_MATMUL:
            assert input_size == 2
            activation = [0]
            assert self.get_parameter_from_pb(pbOp, PMParameter.PM_ACTI, activation)
            opx = self.create_matmul(inputs[0], inputs[1], ActiMode(activation[0]))
        
        elif op_type == OpType.OP_TRANSPOSE:
            pass
        elif op_type in [OpType.OP_POOL2D_MAX, OpType.OP_POOL2D_AVG, OpType.OP_BATCHNORM]:
            pass 
        else: 
            print(op_type)
            assert False
        if opx == None: 
            print(op_type)
        if isSrcOp:
            self.srcOps.append(opx)
        else:
            self.dstOps.append(opx)
    
    def run(self, depth: int, graph, 
            candidates, hashmap: List[int], 
            threshold: float, maxNumOps: int,
            ) -> None:
        from graph import Graph
        assert isinstance(graph, Graph)
        assert isinstance(candidates, list)
        if depth >= len(self.srcOps):
            passTag = True
            for dstOp in self.dstOps:
                if passTag:
                    passTag = passTag and self.create_new_operator(dstOp, dstOp.mapOp)
            if not passTag:
                return
            # Check that output tensors with external edges are mapped
            for mappedOp, mappedOpX in self.mappedOps.items():
                # print(mappedOp, mappedOp.ptr.inputs)
                edge_list = graph.outEdges[mappedOp]
                for e in edge_list:
                    if e.dstOp not in self.mappedOps:
                        # dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
                        srcTen = TensorX()
                        srcTen.op = mappedOpX
                        srcTen.idx = e.srcIdx
                        if srcTen not in self.mappedOutputs:
                            passTag = False
                            return
            self.print_rule()
            # Generate a new graph by applying xfer rule
            new_graph = self.create_new_graph(graph)
            if new_graph.has_loop():
                return
            assert new_graph.check_correctness()
            if new_graph.total_cost() < threshold and new_graph.inEdges.size() < maxNumOps:
                graph_hash_v = new_graph.hash()
                if graph_hash_v  not in hashmap:
                    hashmap.add(graph_hash_v)
                    heapq.heappush(candidates, new_graph) 
                                 
        else:
            srcOp = self.srcOps[depth]
            for op, edge_set in graph.inEdges.items():
                if self.can_match(srcOp, op, graph) and not op in self.mappedOps:
                    # print(f"match: depth {depth}")
                    self.match(srcOp, op, graph)
                    self.run(depth+1, graph, candidates, hashmap, threshold, maxNumOps)
                    # print(f"before unmatch depth: {depth}")
                    self.unmatch(srcOp, op, graph)
    
    def create_new_graph(self, graph): 
        from graph import Graph
        assert isinstance(graph, Graph)
        new_graph = Graph()
        new_graph.subst_history = graph.subst_history
        subst = Graph.GraphSubst()
        
        for i in range(len(self.srcOps)):
            op = self.srcOps[i].mapOp
            subst.srcOps.append(op)
        for i in range(len(self.dstOps)):
            op = self.dstOps[i].mapOp
            subst.dstOps.append(op)
        new_graph.subst_history.append(subst)
        
        # Step 2: add edges to the graph
        for op, edge_list in graph.inEdges.items():
            if op not in self.mappedOps:
                # Unmapped ops
                for e in edge_list:
                    if e.srcOp in self.mappedOps:
                        # mapped src -> unmapped dst
                        srcTen = TensorX()
                        srcTen.op = self.mappedOps[e.srcOp]
                        srcTen.idx = e.srcIdx
                        assert srcTen in self.mappedOutputs
                        dstTen = self.mappedOutputs[srcTen]
                        new_graph.add_edge(dstTen.op.mapOp, e.dstOp, dstTen.idx, e.dstIdx)
                    else:
                        # unmapped src -> unmapped dst
                        new_graph.add_edge(e.srcOp, e.dstOp, e.srcIdx, e.dstIdx)
        # Step 3: add edges for mapped ops
        for dstOp in self.dstOps:
            for i in range(len(dstOp.inputs)):
                if dstOp.inputs[i].op is None:
                    # unmapped src -> mapped dst
                    it = self.mappedInputs.get(dstOp.inputs[i].idx)
                    assert it is not None
                    srcEdge = it[0]
                    new_graph.add_edge(srcEdge[0], dstOp.mapOp, srcEdge[1], i)
                else:
                    # mapped src -> mapped dst
                    srcOp = dstOp.inputs[i].op
                    srcIdx = dstOp.inputs[i].idx
                    new_graph.add_edge(srcOp.mapOp, dstOp.mapOp, srcIdx, i)
        return new_graph
    
    def create_new_operator(self, opx: OpX, op: Op) -> bool:
        # print(opx.type)
        op_type = opx.type
        if op_type == OT.OP_CONV2D:
            assert len(opx.inputs) == 2
            input = opx.inputs[0].to_tensor(self)
            weight = opx.inputs[1].to_tensor(self)
            # print(input, weight)
            strideH, strideW, padding, activation = [0], [0], [0], [0]
            assert opx.get_pm_constraint(PMParameter.PM_STRIDE_H, strideH)
            assert opx.get_pm_constraint(PMParameter.PM_STRIDE_W, strideW)
            assert opx.get_pm_constraint(PMParameter.PM_PAD, padding)
            assert opx.get_pm_constraint(PMParameter.PM_ACTI, activation)
            strideH, strideW, padding, activation = strideH[0], strideW[0], padding[0], activation[0]
            op = self.model.get_or_create_conv2d(input, weight, strideH, strideW, padding, activation)
        elif op_type == OT.OP_EW_ADD or op_type == OT.OP_EW_MUL:
            assert len(opx.inputs) == 2 
            input0 = opx.inputs[0].to_tensor(self)
            input1 = opx.inputs[1].to_tensor(self)
            # print(input0.dim, input1.dim)
            op = self.model.get_or_create_element(op_type, input0, input1)
        elif op_type == OT.OP_CONCAT:
            # TODO: assume don't need copy for now
            inputs: List[Tensor] = [Tensor()] * (MAX_NUM_INPUTS)
            needCopy: List[bool] = [True] * (MAX_NUM_INPUTS)
            for i in range(len(opx.inputs)):
                inputs[i] = opx.inputs[i].to_tensor(self)
                needCopy[i] = False
            axis = [0]
            assert opx.get_pm_constraint(PMParameter.PM_AXIS, axis)
            # TODO: implement get_or_create_concat
        else:
            print(op_type)
            raise NotImplementedError
        # Check operator validness
        if op == Op.INVALID_OP:
            # print("IN_VALID_OP")
            return False
        for tnc in opx.tnConstraints:
            tncResult = self.check_tnConstraints(tnc, op)
            if not tncResult:
                return False
        opx.mapOp = op
        return True
    
    @staticmethod
    def load_graph_xfer_from_pb_file(model: Model, 
                                     xfers: List['GraphXfer'], 
                                     filename: str,
                                     ) -> None:
        # create message object
        collection = rules_pb2.RuleCollection()

        # read the Protobuf message from file
        collection.ParseFromString(open(filename, "rb").read())
        
        # iterate for each rule in the collection
        # for i in range(len(collection.rule)):
        for i in range(133):
            rule = collection.rule[i]
            mappedInputs: Dict[int, TensorX] = dict()
            subst = GraphXfer(model)
            # print(rule)
            for j in range(len(rule.srcOp)):
                subst.create_operator_from_pb(rule.srcOp[j], mappedInputs, True)
            for j in range(len(rule.dstOp)):
                subst.create_operator_from_pb(rule.dstOp[j], mappedInputs, False)
           
            for mapOutput in rule.mappedOutput:   
                srcOpId = mapOutput.srcOpId
                dstOpId = mapOutput.dstOpId
                srcTsId = mapOutput.srcTsId
                dstTsId = mapOutput.dstTsId
                assert srcOpId < len(subst.srcOps)
                assert dstOpId < len(subst.dstOps)
                assert srcTsId < len(subst.srcOps[srcOpId].outputs)
                assert dstTsId < len(subst.dstOps[dstOpId].outputs)
                subst.map_output(subst.srcOps[srcOpId].outputs[srcTsId],
                                 subst.dstOps[dstOpId].outputs[dstTsId])
            xfers.append(subst)
            # subst.print_rule()
    
    
    def print_rule(self):
        print("Src:")
        for i in range(len(self.srcOps)):
            srcOp = self.srcOps[i]
            input_list = []
            for input in srcOp.inputs:
                if input.op != None:
                    input_list.append(f"Op{self.srcOps.index(input.op)}.output[{input.idx}]")
                else: 
                    input_list.append(f"Tensor{input.idx}")
            # print(input_list)
            input_list_str = ", ".join(input_list)
            print(f"\tOp{i}: {srcOp.type} ({input_list_str})")
        print("Dst:")
        for i in range(len(self.dstOps)):
            dstOp = self.dstOps[i]
            input_list = []
            for input in dstOp.inputs:
                if input.op != None:
                    input_list.append(f"Op{self.dstOps.index(input.op)}.output[{input.idx}]")
                else: 
                    input_list.append(f"Tensor{input.idx}")
            # print(input_list)
            input_list_str = ", ".join(input_list)
            print(f"\tOp{i}: {dstOp.type} ({input_list_str})")
        mapped_str_list = []
        for mappedSrcTensorX, mappedDstTensorX in self.mappedOutputs.items():
            assert isinstance(mappedSrcTensorX, TensorX)
            assert isinstance(mappedDstTensorX, TensorX)
            mappedSrc = f"Src.Op{self.srcOps.index(mappedSrcTensorX.op)}.output[{mappedSrcTensorX.idx}]"
            mappedDst = f"Dst.Op{self.dstOps.index(mappedDstTensorX.op)}.output[{mappedDstTensorX.idx}]"
            mapped_str_list.append(f"\t{mappedSrc} -> {mappedDst}")
        print("Mapping:")
        for m in mapped_str_list: print(m)
        print("")
        
def to_tn_parameter(isInput: bool, n: int) -> TNParameter:
    if isInput:
        mapping = {0: TNParameter.IN_0, 1: TNParameter.IN_1, 2: TNParameter.IN_2, 3: TNParameter.IN_3, 4: TNParameter.IN_4, 5: TNParameter.IN_5}
    else:
        mapping = {0: TNParameter.OU_0, 1: TNParameter.OU_1, 2: TNParameter.OU_2, 3: TNParameter.OU_3, 4: TNParameter.OU_4, 5: TNParameter.OU_5}

    assert n in mapping, "Invalid tn_parameter: n should be in the range [0, 5]"
    return mapping[n]

def to_dim_parameter(n: int) -> DIMParameter:
    mapping = {0: DIMParameter.DIM_0, 1: DIMParameter.DIM_1, 2: DIMParameter.DIM_2, 3: DIMParameter.DIM_3}
    assert n in mapping, "Invalid dim_parameter: n should be in the range [0, 3]"
    return mapping[n]
        