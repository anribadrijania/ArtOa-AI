import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import requests
import time
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T

start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

rcnn_model = model = maskrcnn_resnet50_fpn_v2()
model.load_state_dict(torch.load("./maskrcnn_v2.pth", map_location=device))
model.to(device)
model.eval()

url = "https://media.istockphoto.com/id/1008108158/photo/retro-wooden-cabinet-and-a-painting-in-an-empty-living-room-interior-with-white-walls-and.jpg?s=612x612&w=0&k=20&c=vDG7xSaX62DfJ6khAulcHb657M-pA-tlzQcxmm3lpK0="
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image_np = np.array(image)

# Transform the image for model input
transform = T.Compose([T.ToTensor()])
input_tensor = transform(image).unsqueeze(0).to(device)

# Perform segmentation
with torch.no_grad():
    outputs = model(input_tensor)

# Move results to CPU
scores = outputs[0]['scores'].cpu().numpy()
masks = outputs[0]['masks'].squeeze().cpu().numpy()
labels = outputs[0]['labels'].cpu().numpy()

# Define threshold for keeping objects
score_threshold = 0.01

# Create an empty mask for non-wall objects
object_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

for i in range(len(scores)):
    if scores[i] < score_threshold:
        continue  # Skip low-confidence detections

    # Get object mask (threshold it)
    obj_mask = (masks[i] > 0.5).astype(np.uint8) * 255

    # Merge with the main object mask
    object_mask = cv2.bitwise_or(object_mask, obj_mask)

# Convert to RGBA image (add an alpha channel)
rgba_image = np.dstack((image_np, object_mask))

# Save the image with transparency
output_path = "objects_extracted.png"
Image.fromarray(rgba_image).save(output_path)

print("Objects extracted and saved with transparency.")
end = time.time()
print(f"Inference time: {end - start:.3f} seconds")
