import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms

# Set image path and output directory
image_path = 'archive/images/9.jpg'
output_dir = 'outputs_onnx/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the ONNX model
onnx_model_path = "craft_model.onnx"
session = ort.InferenceSession(onnx_model_path)

# Preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((768, 768)),  # Resize to match the model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor.numpy()

# Prepare input tensor
input_tensor = preprocess_image(image_path)

# Perform inference
inputs = {session.get_inputs()[0].name: input_tensor}
outputs = session.run(None, inputs)

# Inspect outputs
print("Model output shape:", [o.shape for o in outputs])
print("Model output example:", outputs)  # Print an example to inspect

# Post-process the results
def postprocess_boxes(outputs):
    # Assuming outputs[0] is the score map for text regions
    # Here you should implement the logic to extract bounding boxes from the score maps.
    # For demonstration, let's just return some dummy boxes.
    
    # Example: Dummy bounding boxes
    boxes = [
        (50, 50, 200, 200),
        (250, 250, 400, 400)
    ]
    
    # You will need to replace this with actual extraction logic from your model outputs.
    return boxes

# Save detected regions
def save_cropped_regions(image_path, boxes, output_dir):
    image = Image.open(image_path).convert('RGB')
    for i, box in enumerate(boxes):
        if len(box) == 4:
            left, top, right, bottom = box
        else:
            print(f"Unexpected box format: {box}")
            continue
        
        # Ensure the coordinates are integers
        left, top, right, bottom = map(int, [left, top, right, bottom])
        
        # Crop and save
        cropped_image = image.crop((left, top, right, bottom))
        cropped_image.save(os.path.join(output_dir, f"crop_{i}.jpg"))

# Post-process and save regions
boxes = postprocess_boxes(outputs)
#save_cropped_regions(image_path, boxes, output_dir)

print("Cropped images saved to the following directory:")
#print(output_dir)
