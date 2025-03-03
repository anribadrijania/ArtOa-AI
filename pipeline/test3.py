from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import torch
import utils
import RCNN
import requests
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rcnn_model = maskrcnn_resnet50_fpn_v2()
rcnn_model.load_state_dict(torch.load("./maskrcnn_v2.pth", map_location=device))
rcnn_model.to(device)
rcnn_model.eval()


def segment_image(segmentor, wall):
    """
    Perform image segmentation on the given wall image.

    :param segmentor: An instance of the Segment class.
    :param wall: The input wall image.
    :return: Tuple (masks, combined_masks, cropped_objects), or (None, None, None) if no objects are detected.
    """
    masks = segmentor.predict_mask_rcnn(wall, 0.01)  # Perform segmentation

    if masks is None or len(masks) == 0:
        return None

    return masks


url = "https://www.muralunique.com/wp-content/uploads/2023/06/1856-50r_stone-texture-wall-charcoal-color.jpg"
wall = Image.open(requests.get(url, stream=True).raw).convert("RGB")

segmentor = RCNN.SegmentRCNN(rcnn_model, device)

masks, cropped_objects = segment_image(segmentor, wall)
output_path = "objects_extracted.png"
Image.fromarray(masks).save(output_path)