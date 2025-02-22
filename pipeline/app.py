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
from fastapi import FastAPI, HTTPException
from ultralytics import YOLO
from typing import List
from logger import data_request

import logging
import generation
import segmentation
import utils
import asyncio
import os
import base64

# Create FastAPI application and load the YOLO model
app = FastAPI()
model = YOLO("segmentation-v1.pt")


# Define segmentation and image generation functions
async def segment_image(segmentor, wall):
    masks = segmentor.predict(wall)
    if masks is None or len(masks) == 0:
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
        data_request(image_url, prompt, tags, box, n)
        wall = await utils.fetch_image(image_url)
        if wall is None:
            logging.error("Invalid image URL or image could not be fetched.")
            raise HTTPException(status_code=400, detail="Invalid image URL or image could not be fetched.")

        box_width, box_height = utils.get_box_coordinates(wall, box)[:2]
        size = utils.get_best_size(box_width, box_height)
        gen_model, quality = "dall-e-2", "standard"
        prompt = prompt + ", " + ", ".join(tags)

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
                wall_art = utils.place_art_in_box(wall, art, box)
                final_images.append(wall_art)
            return final_images

        print("Segmentation completed. Returning cropped images.")
        for art in generated_images:
            wall_art = utils.place_art_in_box(wall, art, box)
            final_image = utils.return_cropped_object(wall_art, cropped_objects, combined_masks)
            final_images.append(utils.image_to_base64(final_image))

        return {"images": final_images}  # Returning JSON-compatible response

    except HTTPException as e:
        logging.error(f"HTTP server error: {str(e)}")
        raise e

    except Exception as e:
        logging.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

