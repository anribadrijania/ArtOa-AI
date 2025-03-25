"""
Description: This file contains the FastAPI application that serves as the main entry point for the pipeline.
The pipeline is responsible for generating art images and placing them on a wall image asynchronously.
The pipeline uses the YOLO model for segmentation and the OpenAI DALL-E model for image generation.
The API key for the OpenAI API is loaded from the environment variables.
"""

# Import required libraries
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from transformers import AutoModelForImageSegmentation
from typing import List
from logger import log_debug, log_error, log_info, log_warning
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from PIL import Image
import numpy as np
import torch
import generation
import segmentation
import utils
import asyncio
import uuid
import time

# Create FastAPI app instance
app = FastAPI()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_debug(f"App is running on {device}...")

remover_model = None
if device == "cuda":
    remover_model = AutoModelForImageSegmentation.from_pretrained("./pretrained", trust_remote_code=True)
    remover_model.load_state_dict(torch.load("remover_v1.pth"))
    torch.set_float32_matmul_precision("highest")
    remover_model.to("cuda").eval().half()
    log_info(f"START: Remover model loaded successfully.")

rcnn_model = maskrcnn_resnet50_fpn_v2()
rcnn_model.load_state_dict(torch.load("./maskrcnn_v2.pth", map_location=device))
rcnn_model.to(device)
rcnn_model.eval()
log_info(f"START: MaskRCNN model loaded successfully.")

# Define the static directory to store images
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

# Mount static files directory for serving generated images
app.mount("/static", StaticFiles(directory="static"), name="static")


async def segment_image(segmentor, wall):
    """
    Perform image segmentation on the given wall image.

    :param segmentor: An instance of the Segment class.
    :param wall: The input wall image.
    :return: Tuple (masks, combined_masks, cropped_objects), or (None, None, None) if no objects are detected.
    """
    log_debug("Segmenting the wall image...")
    masks = segmentor.predict_masks(wall)  # Perform segmentation

    if masks is None or len(masks) == 0:
        log_warning("No objects found during segmentation!")
        return None

    log_info("Segmentation completed successfully.")
    return masks


async def generate_images(generator, n):
    """
    Generate multiple images asynchronously.

    :param generator: An instance of the Generate class.
    :param n: Number of images to generate.
    :return: A list of generated images.
    """
    log_debug("Requesting OpenAI to generate art images...")

    # Use asyncio.gather to run multiple generation requests in parallel
    tasks = [utils.generate_and_fetch(generator) for _ in range(n)]

    return await asyncio.gather(*tasks)


def prompt_engineering(prompt, tags):
    """
    Generates an engineered prompt for a wall art request.

    Parameters:
    prompt (str): The original order text from the client.
    tags (list): A list of artistic styles to be included.

    Returns:
    str: The formatted prompt incorporating the client's request and styles.
    """
    role = "The next provided prompt is a order written by a client who wants to paint art on their wall, " \
           "only consider the art which must be painted and not the details about wall or anything else. " \
           "very very important: Fill the entire artwork and do not create blank areas or borders around the art. "
    if tags:
        styles = "Also use the following styles: " + ", ".join(tags)
    else:
        styles = ""

    engineered_prompt = role + styles + ". The order text is: " + prompt
    return engineered_prompt


@app.post("/generate-on-wall/")
async def main(image_url: str = "",
               prompt: str = "",
               tags: List[str] = None,
               box: List[float] = None,
               n: int = 4):
    """
    API endpoint to generate and place AI-generated art on a wall image.

    :param image_url: URL of the input wall image.
    :param prompt: Text prompt for art generation.
    :param tags: Additional tags to refine image generation.
    :param box: Coordinates defining the area where art should be placed.
    :param n: Number of image variations to generate.
    :return: List of processed images with art placed on the wall.
    """
    start = time.time()

    try:
        # Validate input parameters and log request details
        request_logger(image_url, prompt, tags, box, n)

        log_debug("Processing input data...")

        # Fetch the input wall image
        wall = await utils.fetch_image(image_url)
        if wall is None:
            log_error("Invalid image URL or image could not be fetched.")
            raise HTTPException(status_code=400, detail="Invalid image URL or image could not be fetched.")

        log_info("Wall image fetched successfully.")

        # Extract box dimensions for placing the generated art
        box_width, box_height, x_min, y_min = utils.get_box_coordinates(wall, box)
        log_info("Defined box coordinates successfully.")

        # Determine the best image size for generation
        size = utils.get_best_size(box_width, box_height)

        # Define generation model and parameters
        gen_model, quality = "dall-e-3", "standard"
        prompt = prompt_engineering(prompt, tags)

        log_info(f"Image generation parameters set: model={gen_model}, prompt={prompt}, size={size}, quality={quality}, n={n}")

        # Initialize segmentation and image generation classes
        rcnn_segmentor = segmentation.MaskRCNN(rcnn_model, device)
        generator = generation.Generate(gen_model, prompt, size, quality, 1)

        # Run segmentation and image generation asynchronously
        masks, generated_images = await asyncio.gather(
            segment_image(rcnn_segmentor, wall),
            generate_images(generator, n)
        )

        log_info("Wall segmentation and art generation completed.")

        log_debug("Placing generated art on the wall...")
        final_images = []

        if masks is None:
            # If no segmentation masks are found, place art directly onto the wall
            for art in generated_images:
                # Apply lighting and texture blending
                wall_art_np = utils.apply_lighting_and_texture(wall, art, box)  # box is already in percentage
                wall_art_pil = Image.fromarray(wall_art_np)
                final_images.append(wall_art_pil)

            log_info("Art placed directly on the wall.")
            return final_images

        # If segmentation is successful, overlay generated art while preserving detected objects
        for art in generated_images:
            # Convert wall and art (PIL) to np arrays
            background_np = np.array(wall)
            art_np = np.array(art)

            # Compute box_percent from (x_min, y_min, x_max, y_max)
            h, w_img = background_np.shape[:2]
            box_percent = [x_min / w_img, y_min / h, (x_min + box_width) / w_img, (y_min + box_height) / h]

            # Apply lighting and texture blending
            wall_art = utils.apply_lighting_and_texture(background_np, art_np, box_percent)
            final_image = utils.return_cropped_objects(wall_art, masks)
            log_info("Cropped objects returned successfully.")
            final_images.append(final_image)

        log_info("Art placement on the wall completed.")

        image_urls = []
        # Save final images to the static directory with unique filenames
        for i, img in enumerate(final_images):
            filename = f"{STATIC_DIR}/output_{uuid.uuid4().hex[:8]}.png"
            img.save(filename)
            print(f"Saved: {filename}")
            image_urls.append(f"http://localhost:8000/{filename}")

        end = time.time()
        log_info(f"Request finished in {end - start:.3f} seconds. Response sent to the client.")

        return {"images": image_urls}

    except HTTPException as e:
        log_error(f"HTTP server error: {str(e)}")
        raise e  # Rethrow HTTP exception

    except Exception as e:
        log_error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def request_logger(image_url, prompt, tags, box, n):
    """
    Validate and log incoming requests.

    :raises HTTPException: If input validation fails.
    """
    log_debug("Receiving request...")

    if not image_url:
        raise HTTPException(status_code=400, detail="Empty image URL in the request.")
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt in the request.")
    if not box:
        raise HTTPException(status_code=400, detail="Empty box coordinates in the request.")
    if len(box) != 4:
        raise HTTPException(status_code=400, detail="Invalid number of box coordinates in the request.")

    # Validate box coordinates
    for coord in box:
        if not isinstance(coord, float):
            raise HTTPException(status_code=400, detail="Box coordinates must be float.")
        if coord < 0 or coord > 1:
            raise HTTPException(status_code=400, detail="Invalid coordinates. Values must be between 0 and 1.")

    if box[0] > box[2] or box[1] > box[3]:
        raise HTTPException(status_code=400, detail="Invalid box coordinates. Check order of values.")

    log_info(f"Request validated successfully: image_url={image_url}, prompt={prompt}, tags={tags}, box={box}, n={n}")
