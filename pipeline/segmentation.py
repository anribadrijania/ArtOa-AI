import utils
import torch
import numpy as np
import cv2


class MaskRCNN:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_masks(self, image):
        image_np, input_tensors = utils.transformer_for_rcnn(image, self.device)
        with torch.no_grad():
            outputs = self.model(input_tensors)

        scores = outputs[0]['scores'].cpu().numpy()
        masks = outputs[0]['masks'].squeeze().cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()

        score_threshold = 0.02

        object_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        for i in range(len(scores)):
            if scores[i] < score_threshold:
                continue  # Skip low-confidence detections

            obj_mask = (masks[i] > 0.5).astype(np.uint8) * 255
            obj_mask = cv2.resize(obj_mask, (object_mask.shape[1], object_mask.shape[0]))
            object_mask = cv2.bitwise_or(object_mask, obj_mask)

        rgba_image = np.dstack((image_np, object_mask))
        return rgba_image
