import torch
from craft_text_detector import load_craftnet_model

# Load the CRAFT model
craft_net = load_craftnet_model(cuda=False)  # Change to True if using GPU

# Set the model to evaluation mode
craft_net.eval()

# Define the input tensor shape
input_tensor_detec = torch.randn((1, 3, 768, 768), requires_grad=False)

# Move the tensor to the appropriate device (e.g., CUDA if using GPU)
# Ensure that the model is also on the same device as the input tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor_detec = input_tensor_detec.to(device)
craft_net = craft_net.to(device)

# Define the ONNX export path
onnx_model_path = "craft_model.onnx"

# Export the model to ONNX format
torch.onnx.export(
    craft_net,
    input_tensor_detec,
    onnx_model_path,
    verbose=True,
    opset_version=11,
    do_constant_folding=True,
    export_params=True,
    input_names=["input"],
    output_names=["output", "output1"],
    dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}}
)

print("Model exported to ONNX format.")
