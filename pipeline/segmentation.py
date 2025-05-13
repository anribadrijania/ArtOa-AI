from torchvision import transforms
import torch
import numpy as np
import cv2


def transformer_for_rcnn(image, device):
    image_np = np.array(image)
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensors = transform(image).unsqueeze(0).to(device)

    return image_np, input_tensors


def preprocess_image(image, device):
    """Preprocess the image while maintaining aspect ratio."""
    original_size = image.size  # Save original size (width, height)

    image_size = (1024, 1024)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)
    return input_tensor, original_size  # Return both image and original size


class MaskRCNN:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    async def predict_masks(self, image, bg_mask, threshold=0.01, overlap_thresh=0.5, containment_thresh=0.8):
        image_np, input_tensors = transformer_for_rcnn(image, self.device)
        with torch.no_grad():
            outputs = self.model(input_tensors)

        scores = outputs[0]['scores'].cpu().numpy()
        masks = outputs[0]['masks'].squeeze().cpu().numpy()

        for i in range(len(scores)):
            if scores[i] < threshold:
                continue
            rcnn_mask = (masks[i] > 0.5).astype(np.uint8)

            # Check IoU overlap with BiRefNet
            intersection = np.logical_and(bg_mask, rcnn_mask).sum()
            union = np.logical_or(bg_mask, rcnn_mask).sum()
            iou = intersection / union if union != 0 else 0

            # Check containment: is most of rcnn_mask inside bg_mask?
            rcnn_area = rcnn_mask.sum()
            if rcnn_area == 0:
                continue
            contained_ratio = intersection / rcnn_area

            # Only add object if it’s not overlapping too much and not already covered
            if iou < overlap_thresh or contained_ratio < containment_thresh:
                bg_mask = cv2.bitwise_or(bg_mask, rcnn_mask * 255)
            else:
                continue

        # Ensure RGB base image (strip alpha if needed)
        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]

        final_image_np = np.dstack((image_np, bg_mask))
        return final_image_np


class BgRemover:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    async def predict_masks(self, image, threshold=0.5):
        input_tensor, original_size = preprocess_image(image, self.device)
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        pred = (preds[0].squeeze() > threshold).float()

        # Convert mask to PIL Image & Resize to original dimensions
        mask_pil = transforms.ToPILImage()(pred)
        mask = mask_pil.resize(image.size)

        # Return binary mask as NumPy array
        mask_np = np.array(mask)
        bg_mask = (mask_np > 128).astype(np.uint8)
        bg_mask = bg_mask.astype(np.uint8) * 255
        return bg_mask

