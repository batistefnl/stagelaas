import onnx

model = onnx.load("policy.onnx")

graph = model.graph

print("Input:")
for input_tensor in graph.input:
    print(f" - {input_tensor.name}")
    print(f"Type: {input_tensor.type.tensor_type.elem_type}")
    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"Shape: {shape}\n")

print("\nOutput:")
for output_tensor in graph.output:
    print(f" - {output_tensor.name}")
    print(f"Type: {output_tensor.type.tensor_type.elem_type}")
    shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
    print(f"Shape: {shape}\n")

print(onnx.helper.printable_graph(model.graph))
