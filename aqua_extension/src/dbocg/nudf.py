import onnx
from onnx import helper, TensorProto, numpy_helper
from graph import Graph, Edge
from op import Op
from tensor import Tensor
from utils import *

from db_executor import DBExecutor

import logging
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _add(op: onnx.NodeProto, graph: Graph, tensors: list[Tensor], initializer) -> Tensor:
    inputs = get_inputs(op, graph, tensors, initializer)
    outputs = graph.element(OpType.OP_EW_ADD, inputs[0], inputs[1])
    return outputs

def _avgpool2d(op: onnx.NodeProto, graph: Graph, tensors: list[Tensor], initializer) -> Tensor:
    inputs = get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "AvgPool2D requires exactly one input"
    attrs = parse_attribute(op.attribute)
    kernels = attrs["kernel_shape"]
    strides = attrs["strides"]
    pads = get_conv_pool_pads_attr(attrs)
    
    padding = get_padding_mode(pads)
    activation = get_activation_mode("NONE")
    outputs = graph.pool2d_avg(inputs[0], kernels[0], kernels[1], strides[0], strides[1], padding, activation)
    return outputs

def _constant(op: onnx.NodeProto, graph: Graph, tensors: list[Tensor], initializer) -> Tensor:
    attrs = parse_attribute(op.attribute)
      # TODO: Currently do not support sparse value
    assert "value" in attrs, "Do not support sparse value for Constant"
    tensor = attrs["value"]
    dims = list(tensor.dims)
    outputs = graph.new_weight(dims, tensor)
    return outputs

def _conv2d(op: onnx.NodeProto, graph: Graph, tensors: list[Tensor], initializer) -> Tensor:
    inputs = get_inputs(op, graph, tensors, initializer)
    attrs = parse_attribute(op.attribute)
    if "group" not in attrs:
        group = 1 # default 1
    else:
        group = attrs["group"]
    pads = get_conv_pool_pads_attr(attrs)
    strides = attrs["strides"]
    padding = get_padding_mode(pads)
    activation = get_activation_mode("NONE")

    # _input: inputs[0], _weight: inputs[1], _bias: inputs[2]
    outputs = graph.conv2d(inputs[0], inputs[1], strides[0], strides[1], padding, activation)
    
    if len(inputs) > 2:
        dim = inputs[2].dim[0]
        reshaped_bias = graph.reshape(inputs[2], [1, dim, 1, 1])
        outputs = graph.element(OpType.OP_EW_ADD, outputs, reshaped_bias)
    return outputs

def _gemm(op: onnx.NodeProto, graph: Graph, tensors: list[Tensor], initializer) -> Tensor:
    inputs = get_inputs(op, graph, tensors, initializer)
    attrs = parse_attribute(op.attribute)
    # print(inputs[0].dim, inputs[1].dim)
    if "transA" in attrs and attrs["transA"] == 1:
        inputs[0] = graph.transpose(inputs[0], (1,0), _shuffle=True)
    if "transB" in attrs and attrs["transB"] == 1:
        inputs[1] = graph.transpose(inputs[1], (1,0), _shuffle=True)
    # print(inputs[0].dim, inputs[1].dim)
    outputs = graph.matmul(inputs[0], inputs[1])
    if len(inputs) > 2:
        outputs = graph.element(OpType.OP_EW_ADD, outputs, inputs[2])
    return outputs

def _pad(op: onnx.NodeProto, graph: Graph, tensors: list[Tensor], initializer) -> Tensor:
    inputs = get_inputs(op, graph, tensors, initializer)
    attrs = parse_attribute(op.attribute)
    # TODO: use the shape information from the ONNX runtime to
    # calculate the exact output shape
    # Currently treat pad as a no op
    #assert sum(attrs['pads']) == 0
    return inputs[0]

def _relu(op: onnx.NodeProto, graph: Graph, tensors: list[Tensor], initializer) -> Tensor:
    inputs = get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Relu requires exactly one input"
    outputs = graph.relu(inputs[0])
    return outputs

def _reshape(op: onnx.NodeProto, graph: Graph, tensors: list[Tensor], initializer) -> Tensor:
    inputs = get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2
    shape = list(inputs[1].data_ptr)
    outputs = graph.reshape(inputs[0], shape)
    return outputs

def _input_tensor_name(graph: Graph, inedge: Edge, op: Op):
    intype = graph.get_operator_type(inedge.srcOp)
    if intype == "Input":
        return "input_feature_map"
        # return "data"
    elif intype == "Weight":
        mytype = graph.get_operator_type(op)
        return f"{mytype}{op.guid}_{input_weight_names[mytype][inedge.dstIdx]}"
    else:
        return _output_tensor_name(graph, inedge.srcOp, inedge.srcIdx)

def _output_tensor_name(graph: Graph, op: Op, idx: int):
    op_type = graph.get_operator_type(op)
    return f"{op_type}{op.guid}_fwd{idx}"

# Add all supported operators
dbocg_operators = dict()
dbocg_operators['Add'] = _add
dbocg_operators['AveragePool'] = _avgpool2d
dbocg_operators['Constant'] = _constant
dbocg_operators['Conv'] = _conv2d
dbocg_operators['Gemm'] = _gemm
dbocg_operators['Pad'] = _pad
dbocg_operators['Relu'] = _relu
dbocg_operators['Reshape'] = _reshape

def load_onnx(filename):
    '''
    Load a onnx file and return a Graph

    @params
    filename is a string containing a file name
    @return
    Loaded in-memory Graph
    '''
    graph = Graph()
    model = onnx.load(filename)
    tensors = dict()
    for t in model.graph.input:
        dims = list()
        for d in t.type.tensor_type.shape.dim:
            dims.append(d.dim_value)
        weight_data = None
        for weight in model.graph.initializer:
            if (weight.name == t.name):
                weight_data = weight
        # if dims[0] == 0:
        #     dims[0] = 4
        # We classify an input to be a pure input if we cannot find its weights
        if weight_data is None:
            tensors[t.name] = graph.new_input(dims)
        else:
            tensors[t.name] = graph.new_weight(dims, data=weight_data)
    
     # Add initializers not in the inputs
    for weight in model.graph.initializer:
        if weight.name not in tensors:
            if weight.dims:
                dims = list(weight.dims)
                weight_data = weight
                tensors[weight.name] = graph.new_weight(dims, data=weight_data)

    # Reorder nodes to satisfy data dependencies
    tensor_owner = dict()
    name_to_op = dict()
    idx = 0
    for op in model.graph.node:
        # print(op.name)
        # print(op.input, op.output)
        # Assign a name to the node if empty
        if len(op.name) == 0:
            op.name = op.op_type + '_' + str(idx)
        idx += 1
        name_to_op[op.name] = op
        for output in op.output:
            tensor_owner[output] = op.name   
    out_edges = dict()
    dependents = dict()
    node_list = list() 
    for op in model.graph.node:
        dependents[op.name] = 0
        for input in op.input:
            if input in tensor_owner:
                dependents[op.name] += 1
                input_node = tensor_owner[input]
                if input_node not in out_edges:
                    out_edges[input_node] = list()
                out_edges[input_node].append(op.name)
        if dependents[op.name] == 0:
            node_list.append(op.name)
    idx = 0
    while idx < len(node_list):
        opname = node_list[idx]
        if opname in out_edges:
            for e in out_edges[opname]:
                dependents[e] -= 1
                if dependents[e] == 0:
                    node_list.append(e)
        idx += 1
    assert len(node_list) == len(model.graph.node), "Internal error when reording ONNX operators"
        
    # Add nodes into TASO graph
    cnt = 0
    for opname in node_list:
        op = name_to_op[opname]
        # print(opname)
        # if cnt == 53:
        #     print("hahaha")
        #     break
        # print(cnt, op.op_type, opname)
        cnt += 1
        if op.op_type in dbocg_operators:
            try:
                outputs = dbocg_operators[op.op_type](op, graph, tensors, model.graph.initializer)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                assert len(outputs) == len(op.output), "Number of output tensors mismatch"
                for i in range(len(outputs)):
                    # to do: check out match
                    tensors[op.output[i]] = outputs[i]
            except InputNotFoundError as e:
                logging.error("[load onnx] Cannot find input tensor for operator: name({}) type({}) (Skipped)".format(opname, op.op_type))
                # logging.error("[load_onx] Error message: %s", e)
                continue
        else:
            print("Found unsupported ONNX operator: {} (Skipped)".format(op.op_type))
            continue
    return graph

def optimize(graph: Graph, alpha: float = 1.0, budget: int = 1000, print_subst: bool = False):
    return graph.optimize(alpha, budget, print_subst)


input_weight_names = dict()
input_weight_names['Add'] = ['input1', 'input2']
input_weight_names['AveragePool'] = ['input']
input_weight_names['BatchNormalization'] = ['input', 'scale', 'bias', 'mean', 'var']
input_weight_names['Concat'] = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6']
input_weight_names['Conv'] = ['input', 'weight', 'bias']
input_weight_names['MatMul'] = ['input', 'weight']
input_weight_names['Mul'] = ['input1', 'input2']
input_weight_names['Reshape'] = ['input', 'shape']
input_weight_names['BroadcastAdd'] = ['input1', 'input2']
input_weight_names['Transpose'] = ['input']

class nUDF():
    def __init__(self, name, path):
        self.name = name
        self.input_type = "virtual"
        path = "/home/pyc/workspace/aqua/aqua_extension/res18.onnx"
        self.__db_executor = DBExecutor()
        self.graph = load_onnx(path)
    
    def export(self):
        graph = self.graph
        opList = graph.get_operator_list()
        # graph_nodes = list()
        # graph_inputs = list()
        # graph_initializers = list()
        graph_outputs = list()
        output_guids = dict()
        table_mapping_weight = dict()
        
        skip_nodes = dict()

        nudf_body = (
                "WITH\n"
                "  input_feature_map AS (\n"
                f"\tSELECT id, im2col_init(c.image) AS value\n"
                "\t)"
            )
        
        del opList[-1]
        for op in opList:
            mytype = graph.get_operator_type(op)
            inedges = graph.get_input_edges(op)
            inputs = list()
            # print(op, mytype)
            for e in inedges:
                intype = graph.get_operator_type(e.srcOp)
                in_name = _input_tensor_name(graph, e, op)
                # print(f"[input]{in_name}")
                inputs.append(in_name)
                output_guids.pop((e.srcOp.guid, e.srcIdx), None)
                # output_guids.pop((e['srcOp']['guid'], e['srcIdx']), None)
                if intype == 'Input' or intype == 'Weight':
                    pass 
                if intype == 'Weight':
                    weight_dims = graph.get_input_dims(op, e.dstIdx)
                    weight_val = graph.get_weight_value(e.srcOp)
                    if mytype == "Conv" or mytype == "Transpose":
                        weight_val = weight_val.reshape(weight_val.shape[0], -1)
                        weight_val_array = weight_val.tolist()
                        table_mapping_weight[in_name] = weight_val_array
            outputs = list()
            for i in range(op.ptr.numOutputs):
                outputs.append(_output_tensor_name(graph, op, i))
                output_guids[(op.guid, i)] = op
            # print(mytype)
            exe_cmd = ""
            if mytype == "Conv":
                kernels = graph.get_operator_attr(op, "kernel_shape")
                strides = graph.get_operator_attr(op, "strides")
                paddings = graph.get_operator_attr(op, "pads")
                if inputs[0] == "input_feature_map":
                    exe_cmd = f"SELECT id, kfm(W.val, I.value) AS value" \
                            + f"\n\tFROM {inputs[0]} I, {inputs[1]} W"
                else:
                    exe_cmd = f"SELECT id, kfm(W.val, im2col(I.value, kernel:={kernels[0]}, padding:={paddings[0]}, stride:={strides[0]})) AS value" \
                            + f"\n\tFROM {inputs[0]} I, {inputs[1]} W"
            elif mytype == "Relu":
                inputs[0] = skip_nodes.get(inputs[0], inputs[0])
                exe_cmd = f"SELECT id, relu(I.value) AS value\n\tFROM {inputs[0]} I"
            elif mytype == "Add":
                if inputs[1].startswith("Reshape"):
                    exe_cmd = ""
                    skip_nodes[_output_tensor_name(graph, op, 0)] = inputs[0]
                else:
                    inputs[0] = skip_nodes.get(inputs[0], inputs[0])
                    inputs[1] = skip_nodes.get(inputs[1], inputs[1])
                    exe_cmd = f"SELECT I1.id, madd(I1.value, I2.value) AS value" \
                        + f"\n\tFROM {inputs[0]} I1\n\tINNER JOIN {inputs[1]} I2 on I1.id=I2.id"
            elif mytype == "MatMul":
                inputs[1] = skip_nodes.get(inputs[1], inputs[1])
                exe_cmd = f"SELECT id, mvm(W.val, I.value) AS value" \
                        + f"\n\tFROM {inputs[0]} I, {inputs[1]} W"
            elif mytype == "AveragePool":
                kernels = graph.get_operator_attr(op, "kernel_shape")
                exe_cmd = f"SELECT id, avgpool(I.value, kernel:={kernels[0]}) AS value" \
                        + f"\n\tFROM {inputs[0]} I"
            elif mytype == "Reshape":
                shape = graph.get_output_dims(op, 0)
                if len(shape) != 2:
                    continue
                exe_cmd = f"SELECT id, array_agg(t.value) as value" \
                        + f"\n\tFROM (SELECT id, unnest(I.value) as value FROM {inputs[0]} I) AS t" \
                        + f"\n\tGROUP BY id"
            elif mytype == "Transpose":
                exe_cmd = ""
                skip_nodes[_output_tensor_name(graph, op, 0)] = inputs[0]
            else: 
                exe_cmd = ""
            if exe_cmd != "":
                nudf_body += f",\n  {outputs[0]} AS (\n\t{exe_cmd}\n\t)"
            # print(f"{outputs[0]} AS (\n\t{exe_cmd}\n\t)")
        for guid, idx in output_guids:
            op = output_guids[(guid, idx)]
            graph_outputs.append(_output_tensor_name(graph, op, idx))
            
        nudf_body += (
                "\nSELECT l.name AS res FROM\n"
                "cifar10_labels l\n"
                "JOIN (\n"
                f"\tSELECT argmax(value) AS label FROM {graph_outputs[0]}) t\n"
                "\tON t.label = l.label"
            )
        # print(table_mapping_weight.keys())
        nudf_file = "/home/pyc/workspace/aqua_v2/aqua_extension/src/tool_sql/nudf.sql"
        with open(nudf_file, 'a+') as file:
            file.write(nudf_body)
        nudf_command = nudf_body 
        
        # self.__db_executor.write_tensor_parallel(table_mapping_weight)
        return nudf_command
    
    

if __name__ == "__main__":
    setup_logging()
    # old_graph = load_onnx("/home/pyc/workspace/aqua/aqua_extension/res18.onnx")
    # export(old_graph)
    # new_graph = optimize(old_graph)
    t = nUDF("color", "test.onnx")
    nudf_cmd = t.export()
    # print(nudf_cmd)
    
    # import numpy as np
    # graph = Graph()
    # input = graph.new_input(dims = [1,3,32,32])
    # weight1 = graph.new_weight([64,3,3,3], np.random.rand(64,3,3,3))
    # t1 = graph.conv2d(
    #     _input=input, _weight=weight1, _strideH=1, _strideW=1,
    #     _padding=PaddingMode.PD_MODE_SAME, _activation=ActiMode.AC_MODE_NONE)
    # t2 = graph.relu(t1)
    # t3 = graph.pool2d_avg(t2, 2, 2, 2, 2, PaddingMode.PD_MODE_VALID, ActiMode.AC_MODE_NONE)
    # new_graph = optimize(graph)

