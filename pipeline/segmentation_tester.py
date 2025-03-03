# Import required libraries
from fastapi import FastAPI, HTTPException
from pathlib import Path
from logger import log_debug, log_error, log_info, log_warning
import uuid
import time
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import torch
import utils
import RCNN
import requests
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

rcnn_model = maskrcnn_resnet50_fpn_v2()
rcnn_model.load_state_dict(torch.load("./maskrcnn_v2.pth", map_location=device))
rcnn_model.to(device)
rcnn_model.eval()

# Define the static directory to store images
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist


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


def main(url, box):
    start = time.time()
    try:
        # Fetch the input wall image
        wall = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        if wall is None:
            raise HTTPException(status_code=400, detail="Invalid image URL or image could not be fetched.")
        print(1)

        # Extract box dimensions for placing the generated art
        box_width, box_height, x_min, y_min = utils.get_box_coordinates(wall, box)
        print(2)

        # Initialize segmentation and image generation classes
        segmentor = RCNN.SegmentRCNN(rcnn_model, device)
        print(3)

        masks = segment_image(segmentor, wall)
        print(4)

        art_url = "https://thevirtualinstructor.com/blog/wp-content/uploads/2013/08/understanding-abstract-art.jpg"
        art = Image.open(requests.get(art_url, stream=True).raw).convert("RGB")
        final_images = []
        print(5)

        if masks is None:
            # If no segmentation masks are found, place art directly onto the wall
            wall_art = utils.place_art_in_box(wall, art, box_width, box_height, x_min, y_min)
            final_images.append(wall_art)
            return final_images
        print(6)

        # If segmentation is successful, overlay generated art while preserving detected objects
        wall_art = utils.place_art_in_box(wall, art, box_width, box_height, x_min, y_min)
        print(7)
        final_image = utils.return_cropped_objects(wall_art, masks)
        print(8)
        final_image.save("output.png")
        print(9)

        end = time.time()
        print(f"Inference time: {end - start:.3f} seconds")

        return None

    except HTTPException as e:
        raise e  # Rethrow HTTP exception

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


url = "https://www.muralunique.com/wp-content/uploads/2023/06/1856-50r_stone-texture-wall-charcoal-color.jpg"
box = [0.1, 0.2, 0.6, 0.8]
main(url, box)
