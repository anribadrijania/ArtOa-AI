import torch
from PIL import Image
import numpy as np
from torchvision import transforms as T
import cv2


class SegmentRCNN:
    """
    A class for performing image segmentation using a provided model.
    """
    def __init__(self, model, device):
        """
        Initialize the Segment class with a segmentation model.

        :param model: The pre-trained segmentation model to use for prediction.
        """
        self.model = model
        self.device = device
        self.transform = T.Compose([T.ToTensor()])

    def predict_mask_rcnn(self, image, conf_threshold=0.5):
        """
        Perform segmentation on the given image.

        :param image: The input image (PIL Image or NumPy array).
        :param conf_threshold: Confidence threshold for filtering detections.
        :return: A NumPy array of segmentation masks if available, otherwise None.
        """
        image_np = np.array(image)

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        scores = outputs[0]['scores'].cpu().numpy()
        masks = outputs[0]['masks'].squeeze().cpu().numpy() if outputs[0]['masks'] is not None else None

        object_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

        for i in range(len(scores)):
            if scores[i] < conf_threshold:
                continue  # Skip low-confidence detections

            # Get object mask (threshold it)
            obj_mask = (masks[i] > 0.5).astype(np.uint8) * 255

            # Merge with the main object mask
            object_mask = cv2.bitwise_or(object_mask, obj_mask)

        # Convert to RGBA image (add an alpha channel)
        rgba_image = np.dstack((image_np, object_mask))

        return rgba_image
