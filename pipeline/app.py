"""
Description: This file contains the FastAPI application that serves as the main entry point for the pipeline.
The pipeline is responsible for generating art images and placing them on a wall image asynchronously.
The pipeline uses the YOLO model for segmentation and the OpenAI DALL-E model for image generation.
The API key for the OpenAI API is loaded from the environment variables.

The pipeline consists of the following main components:
    1. FastAPI Application: Defines the main FastAPI application that handles incoming HTTP requests.
    2. Middleware: Logs incoming requests and their status codes.
    3. Segmentation: Defines the Segment class that uses the YOLO model for image segmentation.
    4. Generation: Defines the Generate class that uses the OpenAI API for image generation.
    5. Utils: Contains utility functions for image processing and manipulation.
    6. Logger: Configures the logging for the pipeline.

The FastAPI application defines the following endpoints:
1. POST /generate-on-wall/: Accepts a JSON payload with the following parameters:
    - image_url: URL of the wall image.
    - prompt: Prompt for image generation.
    - tags: List of tags for the prompt.
    - box: List of box coordinates for placing the art on the wall.
    - n: Number of art variations to generate.
    The endpoint processes the request by segmenting the wall image, generating art images, and placing them on the wall.

----------------------------------------------------------------------------------------------------------------------------
To run the pipline, you need to set the OPENAI_API_KEY environment variable in .env file with your OpenAI API key.
Then you can run the FastAPI application using the following command in the pipeline directory:
uvicorn app:app --reload
----------------------------------------------------------------------------------------------------------------------------
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

# Create FastAPI application and load the YOLO model
app = FastAPI()
model = YOLO("segmentation-v1.pt")

# Create static directory if it doesn't exist
STATIC_DIR = Path("static/images")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Define segmentation and image generation functions
async def segment_image(segmentor, wall):
    log_debug("Segmenting the wall image...")
    masks = segmentor.predict(wall)
    if masks is None or len(masks) == 0:
        log_warning("No learned objects found!")
        return None, None, None
    combined_masks = utils.combine_masks(wall, masks)
    cropped_objects = utils.crop_object_with_mask(wall, combined_masks)
    return masks, combined_masks, cropped_objects


async def generate_images(generator, n):
    tasks = [utils.generate_and_fetch(generator) for _ in range(n)]
    return await asyncio.gather(*tasks)


# Define the main endpoint for generating art on a wall image
@app.post("/generate-on-wall/")
async def main(image_url: str = "",
               prompt: str = "",
               tags: List[str] = None,
               box: List[float] = None,
               n: int = 4):
    try:
        request_logger(image_url, prompt, tags, box, n)

        log_debug("Processing data...")
        wall = await utils.fetch_image(image_url)
        if wall is None:
            log_error("Invalid image URL or image could not be fetched.")
            raise HTTPException(status_code=400, detail="Invalid image URL or image could not be fetched.")
        log_info("Wall image received and was fetched successfully.")

        box_width, box_height, x_min, y_min = utils.get_box_coordinates(wall, box)
        log_info("Box was defined successfully.")

        size = utils.get_best_size(box_width, box_height)
        gen_model, quality = "dall-e-2", "standard"
        prompt = prompt + ", " + ", ".join(tags)
        log_info(f"Generator parameters were set successfully: gen_model={gen_model}, prompt={prompt}, size={size}, quality={quality}, n={n}")

        segmentor = segmentation.Segment(model)
        generator = generation.Generate(gen_model, prompt, "256x256", quality, n)

        (masks, combined_masks, cropped_objects), generated_images = await asyncio.gather(
            segment_image(segmentor, wall),
            generate_images(generator, n)
        )
        print(generated_images)

        final_images = []
        if masks is None:
            logging.info("No segmentation masks found. Placing art directly.")
            for art in generated_images:
                wall_art = utils.place_art_in_box(wall, art, box_width, box_height, x_min, y_min)
                final_images.append(wall_art)
            return final_images

        print("Segmentation completed. Returning cropped images.")
        for art in generated_images:
            wall_art = utils.place_art_in_box(wall, art, box_width, box_height, x_min, y_min)
            final_image = utils.return_cropped_object(wall_art, cropped_objects, combined_masks)
            final_images.append(final_image)

        for i, img in enumerate(final_images):
            filename = f"{STATIC_DIR}output_{uuid.uuid4().hex[:8]}.png"  # Unique filename
            img.save(filename)
            print(f"Saved: {filename}")
        # return response.json()  # Returning JSON-compatible response

    except HTTPException as e:
        logging.error(f"HTTP server error: {str(e)}")
        raise e

    except Exception as e:
        logging.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# @app.post("/upload-images/")
# async def upload_images(images: list[UploadFile]):
#     image_urls = []
#
#     for img in images:
#         # Save image with a unique name
#         ext = img.filename.split(".")[-1]
#         file_name = f"{uuid.uuid4()}.{ext}"
#         file_path = STATIC_DIR / file_name
#
#         # Save the image
#         with open(file_path, "wb") as buffer:
#             buffer.write(await img.read())
#
#         # Generate URL
#         image_url = f"http://127.0.0.1:8000/static/images/{file_name}"
#         image_urls.append(image_url)
#
#     return {"image_urls": image_urls}

def request_logger(image_url, prompt, tags, box, n):
    log_debug(f"Receiving request...")
    if image_url == "":
        log_error("Empty image URL.")
        raise HTTPException(status_code=400, detail="Empty image URL in the request.")
    if prompt == "":
        log_error("Empty prompt.")
        raise HTTPException(status_code=400, detail="Empty prompt in the request.")
    if box is []:
        log_error("Empty box coordinates.")
        raise HTTPException(status_code=400, detail="Empty box coordinates in the request.")
    if len(box) != 4:
        log_error("Invalid number of box coordinates.")
        raise HTTPException(status_code=400, detail="Invalid number of box coordinates in the request.")
    for coord in box:
        if not isinstance(coord, float):
            log_error("Invalid box coordinates.")
            raise HTTPException(status_code=400, detail="Box coordinates must be float.")
        if coord > 1:
            log_error("Bigger than 1 coordinates")
            raise HTTPException(status_code=400, detail="Invalid coordinates. Coordinates can't be bigger than 1.")
        elif coord < 0:
            log_error("Smaller than 0 coordinates")
            raise HTTPException(status_code=400, detail="Invalid coordinates. Coordinates can't be smaller than 0.")
    if box[0] > box[2] or box[1] > box[3]:
        log_error("Invalid box coordinates.")
        raise HTTPException(status_code=400, detail="Invalid box coordinates.")

    log_info(f"Request received successfully: image_url={image_url}, prompt={prompt}, tags={tags}, box={box}, n={n}")