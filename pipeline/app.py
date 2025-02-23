"""
Description: This file contains the FastAPI application that serves as the main entry point for the pipeline.
The pipeline is responsible for generating art images and placing them on a wall image asynchronously.
The pipeline uses the YOLO model for segmentation and the OpenAI DALL-E model for image generation.
The API key for the OpenAI API is loaded from the environment variables.
"""

# Import required libraries
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from ultralytics import YOLO
from typing import List
from logger import log_debug, log_error, log_info, log_warning
import requests
import logging
import generation
import segmentation
import utils
import asyncio
import os
import io
import uuid

# Create FastAPI app instance
app = FastAPI()

# Load the YOLO segmentation model
model = YOLO("segmentation-v1.pt")

# Define the static directory to store images
STATIC_DIR = Path("static/images")
STATIC_DIR.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

# Mount static files directory for serving generated images
app.mount("/static", StaticFiles(directory="static/images"), name="static")


async def segment_image(segmentor, wall):
    """
    Perform image segmentation on the given wall image.

    :param segmentor: An instance of the Segment class.
    :param wall: The input wall image.
    :return: Tuple (masks, combined_masks, cropped_objects), or (None, None, None) if no objects are detected.
    """
    log_debug("Segmenting the wall image...")
    masks = segmentor.predict(wall)  # Perform segmentation

    if masks is None or len(masks) == 0:
        log_warning("No objects found during segmentation!")
        return None, None, None

    log_info("Segmentation completed successfully.")

    # Combine all detected masks into a single mask
    combined_masks = utils.combine_masks(wall, masks)

    # Crop objects from the image using the combined mask
    cropped_objects = utils.crop_object_with_mask(wall, combined_masks)

    log_info("Masked objects cropped successfully.")
    return masks, combined_masks, cropped_objects


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
    role = "The next provided prompt is a order written by a client who wants to paint art on their wall, " \
           "only consider the art which must be painted and not the details about wall or anything else. " \
           "Fill the entire artwork and do not create blank areas. "
    styles = "Also use the following styles: " + ", ".join(tags)
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
        segmentor = segmentation.Segment(model)
        generator = generation.Generate(gen_model, prompt, size, quality, 1)

        # Run segmentation and image generation asynchronously
        (masks, combined_masks, cropped_objects), generated_images = await asyncio.gather(
            segment_image(segmentor, wall),
            generate_images(generator, n)
        )

        log_info("Wall segmentation and art generation completed.")

        log_debug("Placing generated art on the wall...")
        final_images = []

        if masks is None:
            # If no segmentation masks are found, place art directly onto the wall
            for art in generated_images:
                wall_art = utils.place_art_in_box(wall, art, box_width, box_height, x_min, y_min)
                final_images.append(wall_art)
            log_info("Art placed directly on the wall.")
            return final_images

        # If segmentation is successful, overlay generated art while preserving detected objects
        for art in generated_images:
            wall_art = utils.place_art_in_box(wall, art, box_width, box_height, x_min, y_min)
            final_image = utils.return_cropped_object(wall_art, cropped_objects, combined_masks)
            log_info("Cropped objects returned successfully.")
            final_images.append(final_image)

        log_info("Art placement on the wall completed.")

        image_urls = []
        # Save final images to the static directory with unique filenames
        for i, img in enumerate(final_images):
            filename = f"{STATIC_DIR}/output_{uuid.uuid4().hex[:8]}.png"
            img.save(filename)
            print(f"Saved: {filename}")
            image_urls.append(f"/static/{filename}")

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
