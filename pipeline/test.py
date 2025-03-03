from transformers import AutoModelForImageSegmentation
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import requests
import matplotlib.pyplot as plt
import time
start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
remover_model = AutoModelForImageSegmentation.from_pretrained("./pretrained", trust_remote_code=True)
remover_model.load_state_dict(torch.load("remover_v1.pth"))
torch.set_float32_matmul_precision("highest")
remover_model.to("cuda").eval().half()

def preprocess_image(image, device):
    """Preprocess the image while maintaining aspect ratio."""
    original_size = image.size  # Save original size (width, height)

    image_size = (1024, 1024)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor, original_size  # Return both image and original size


def extract_object(model, image, threshold=0.5):
    input_tensor, original_size = preprocess_image(image, device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    pred = (preds[0].squeeze() > threshold).float()

    # Convert mask to PIL Image & Resize to original dimensions
    mask_pil = transforms.ToPILImage()(pred)
    mask = mask_pil.resize(image.size)

    # Apply mask to the original image
    image.putalpha(mask)
    return image


url = "https://www.muralunique.com/wp-content/uploads/2023/06/1856-50r_stone-texture-wall-charcoal-color.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# image = Image.open('1.jpg').convert("RGB")
threshold = 0.5
final_image = extract_object(model, image, threshold)
final_image.save("output.png")

end = time.time()
print(f"Inference time: {end - start:.3f} seconds")
