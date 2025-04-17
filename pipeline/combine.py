import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests

import time

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Using device: {device}")

# Load the model structure
rcnn_model = maskrcnn_resnet50_fpn_v2()

# Load the locally saved weights
rcnn_model.load_state_dict(torch.load("maskrcnn_v2.pth", map_location=device))

# Move the model to GPU if available
rcnn_model.to(device)
rcnn_model.eval()
print("Model loaded successfully from local storage.")


def predict_masks(model, image):
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
    score_threshold = 0.1

    # Create an empty mask for non-wall objects
    object_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue  # Skip low-confidence detections

        # Get object mask (threshold it)
        obj_mask = (masks[i] > 0.3).astype(np.uint8) * 255

        # Merge with the main object mask
        object_mask = cv2.bitwise_or(object_mask, obj_mask)

    # Convert to RGBA image (add an alpha channel)
    rgba_image = np.dstack((image_np, object_mask))
    return rgba_image

# Imports
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import requests

import time

# Load BiRefNet with weights
from transformers import AutoModelForImageSegmentation
remover_model = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)

# Load quantized model
remover_model.load_state_dict(torch.load("segmentation.pth"))

torch.set_float32_matmul_precision(['highest'][0])
remover_model.to('cuda')
remover_model.eval().half()

def preprocess_image(image):
    """Preprocess the image while maintaining aspect ratio."""
    original_size = image.size  # Save original size (width, height)

    image_size = (1024, 1024)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to("cuda").half()
    return input_tensor, original_size  # Return both image and original size


def extract_object(model, image, threshold=0.5):
    input_tensor, original_size = preprocess_image(image)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    pred = (preds[0].squeeze() > threshold).float()

    # Convert mask to PIL Image & Resize to original dimensions
    mask_pil = transforms.ToPILImage()(pred)
    mask = mask_pil.resize(image.size)

    # Apply mask to the original image
    image.putalpha(mask)
    return image

url = "https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdXB3azYxODUzODk5LXdpa2ltZWRpYS1pbWFnZS1rb3djc3dpei5qcGc.jpg"

if __name__ == "__main__":
    start = time.time()

    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    final_image = predict_masks(rcnn_model, image)

    output_path = "rcnn.png"
    Image.fromarray(final_image).save(output_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(final_image)
    axes[1].set_title("Segmented Image")
    axes[1].axis("off")

    plt.show()

    end = time.time()
    print(f"Inference time: {end - start:.3f} seconds")

    start = time.time()

    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # image = Image.open("/content/1.webp")
    image2 = image.copy()

    threshold = 0.5
    output_image = extract_object(remover_model, image, threshold)
    output_image.save("remover.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(image2)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(output_image)
    axes[1].set_title("Removed Background")
    axes[1].axis("off")

    plt.show()

    end = time.time()
    print(f"Inference time: {end - start:.3f} seconds")