"""
Description: This file contains the FastAPI application that serves as the main entry point for the pipeline.
It includes endpoints to generate AI art or upload custom art and place it on a wall image.
The pipeline uses the YOLO model for segmentation and the OpenAI DALL-E model for image generation.
"""

# Import required libraries
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
# from logger import log_debug, log_info, log_warning, log_error
from transformers import AutoModelForImageSegmentation
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from fastapi.responses import StreamingResponse
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from PIL import Image
from typing import List
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import os
import torch
import utils
import segmentation
import generation
import asyncio
import traceback
import io
import zipfile


# Load environment variables
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
APP_API_KEY = os.getenv("APP_API_KEY")
if not AZURE_ENDPOINT:
    raise ValueError("AZURE_ENDPOINT is not set!")
if not AZURE_OPENAI_API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY is not set!")
if not APP_API_KEY:
    raise ValueError("APP_API_KEY is not set!")

# OpenAI image client
image_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01")

# OpenAI image client
text_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-12-01-preview")

# Define device
device = "cpu"
# log_debug(f"App is running on {device}...")

# Load segmentation models
remover_model = AutoModelForImageSegmentation.from_pretrained("./pretrained", trust_remote_code=True)
remover_model.load_state_dict(torch.load("remover_v2.pth", map_location=torch.device('cpu')))
remover_model.eval()
# log_info("Remover model loaded.")

rcnn_model = maskrcnn_resnet50_fpn_v2()
rcnn_model.load_state_dict(torch.load("./maskrcnn_v2.pth", map_location=torch.device('cpu')))
rcnn_model.eval()
# log_info("MaskRCNN model loaded.")


# Request schema
class GenerateRequest(BaseModel):
    api_key: str
    image_url: str
    prompt: str
    box: List[float]
    tags: List[str] = [""]
    n: int = 4


class CustomRequest(BaseModel):
    api_key: str
    wall_image: str
    box: List[float]
    image_urls: List[str]


class GenerateArtRequest(BaseModel):
    api_key: str
    prompt: str
    tags: List[str] = [""]
    n: int = 4

# Create FastAPI app instance
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate requests
def validate_request(api_key, box):
    """
    Validates the API key and the bounding box format.
    Ensures that the API key matches the expected one and that the box is a list of 4 float values between 0 and 1.
    Raises HTTP exceptions for invalid input.
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is required.")
    if api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    if not box or len(box) != 4:
        raise HTTPException(status_code=400, detail="Box must be 4 float values between 0 and 1.")
    for val in box:
        if not isinstance(val, float) or val < 0 or val > 1:
            raise HTTPException(status_code=400, detail="Invalid box coordinates.")
    if box[0] > box[2] or box[1] > box[3]:
        raise HTTPException(status_code=400, detail="Invalid box order.")


@app.middleware("http")
async def add_logging_and_error_handling(request: Request, call_next):
    """
    Middleware for logging all incoming requests and catching unhandled exceptions.
    Logs HTTP method and URL, and gracefully handles errors by returning appropriate HTTP status codes.
    """
    try:
        response = await call_next(request)
        return response
    except HTTPException as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=e)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=e)


# Helper: segmentation
async def segment_image(rcnn_segmentor, remover_segmentor, wall, box):
    """
    Performs object segmentation on the input wall image.
    Combines the background remover and Mask R-CNN predictions to get final object masks.
    Returns None if no objects are found.
    """
    remover_mask = await remover_segmentor.predict_masks(wall, 0.9)
    final_masks = await rcnn_segmentor.predict_masks(wall, remover_mask, 0.5, box=box)
    if final_masks is None or final_masks.size == 0:
        # log_warning("No objects found during segmentation.")
        return np.zeros_like(wall)
    return final_masks


# Helper: generation
async def generate_images(generator, n):
    """
    Generates `n` AI art images asynchronously using the provided generator instance.
    Returns a list of generated PIL Image objects.
    """
    tasks = [utils.generate_and_fetch(generator) for _ in range(n)]
    return await asyncio.gather(*tasks)


# Helper: process art on wall
def process_wall_and_arts(wall, arts, box, masks):
    """
    Processes the wall image by placing each art image in the specified box region.
    Applies lighting and texture blending, then composites with segmented objects.
    Returns a list of final processed images.
    """
    box_width, box_height, x_min, y_min = utils.get_box_coordinates(wall, box)
    final_images = []
    for art in arts:
        background_np = np.array(wall)
        art_np = np.array(art)
        h, w_img = background_np.shape[:2]
        box_percent = [x_min / w_img, y_min / h, (x_min + box_width) / w_img, (y_min + box_height) / h]
        wall_art = utils.apply_lighting_and_texture(background_np, art_np, box_percent)
        final = utils.return_cropped_objects(wall_art, masks) if masks is not None else Image.fromarray(wall_art)
        final_images.append(final)
    return final_images


# Helper: image response
def create_zip_from_images(images: List[Image.Image]) -> io.BytesIO:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i, img in enumerate(images):
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            zip_file.writestr(f"image_{i}.png", img_bytes.read())
    zip_buffer.seek(0)
    return zip_buffer


@app.post("/generate-on-wall/")
async def generate_on_wall(req: GenerateRequest):
    """
    Endpoint to generate AI art and place it on a user-provided wall image.
    - Validates request.
    - Downloads and processes wall image.
    - Uses DALL-E to generate art.
    - Places each generated art on the wall with visual enhancements.
    Returns a list of image byte arrays.
    """
    now = datetime.now()
    print("Current Time:", now.strftime("%H:%M:%S"))
    validate_request(req.api_key, req.box)
    wall = await utils.fetch_image(req.image_url)
    if wall is None:
        raise HTTPException(status_code=400, detail="Invalid image URL")

    box_width, box_height, *_ = utils.get_box_coordinates(wall, req.box)
    size = utils.get_best_size(box_width, box_height)

    # prompt engineering
    prompt = utils.prompt_engineering(req.prompt, req.tags)
    text_generator = generation.GeneratePrompt(text_client)
    final_prompt = await text_generator.generate_prompt(prompt)

    now = datetime.now()
    print("Current Time:", now.strftime("%H:%M:%S"))


    generator = generation.GenerateImage(image_client, "dall-e-3", final_prompt, size, "standard", "natural", 1)
    segmentors = (
        segmentation.MaskRCNN(rcnn_model, device),
        segmentation.BgRemover(remover_model, device)
    )

    arts_task = asyncio.create_task(generate_images(generator, req.n))
    segment_wall_task = asyncio.create_task(
        segment_image(*segmentors, wall, req.box)
    )

    arts, masks = await asyncio.gather(arts_task, segment_wall_task)
    final_images = process_wall_and_arts(wall, arts, req.box, masks)

    zip_buffer = create_zip_from_images(final_images)
    now = datetime.now()

    # Print the time in HH:MM:SS format
    print("Current Time:", now.strftime("%H:%M:%S"))
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": "attachment; filename=images.zip"
    })


@app.post("/custom-on-wall/")
async def custom_on_wall(req: CustomRequest):
    """
    Endpoint to place user-provided art images onto a wall image.
    - Validates request.
    - Downloads and processes wall image.
    - Downloads all provided art images.
    - Uses segmentation to enhance placement (e.g., background removal, occlusion handling).
    Returns a ZIP file containing the final composited images.
    """
    validate_request(req.api_key, req.box)

    # Fetch wall image
    wall = await utils.fetch_image(req.wall_image)
    if wall is None:
        raise HTTPException(status_code=400, detail="Invalid wall image URL")

    # Fetch all art images from URLs
    arts = []
    for url in req.image_urls:
        art = await utils.fetch_image(url)
        if art is None:
            raise HTTPException(status_code=400, detail=f"Invalid art image URL: {url}")
        arts.append(art.convert("RGB"))

    # Initialize segmentors
    segmentors = (
        segmentation.MaskRCNN(rcnn_model, device),
        segmentation.BgRemover(remover_model, device)
    )

    # Process images and return result
    final_images = await process_wall_and_arts(wall, arts, req.box, segmentors)

    zip_buffer = create_zip_from_images(final_images)
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": "attachment; filename=images.zip"
    })


@app.post("/generate-art/")
async def generate_art(req: GenerateArtRequest):
    """
    Endpoint to generate AI art only (no wall placement).
    - Validates API key and box format.
    - Builds the prompt from user input.
    - Generates n art images using DALL-E.
    - Returns them in a ZIP file.
    """
    from datetime import datetime
    now = datetime.now()

    # Print the time in HH:MM:SS format
    print("Current Time:", now.strftime("%H:%M:%S"))
    # Validate API key only (box is not needed but used in schema)
    if not req.api_key:
        raise HTTPException(status_code=401, detail="API key is required.")
    if req.api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    # prompt engineering
    prompt = utils.prompt_engineering(req.prompt, req.tags)
    text_generator = generation.GeneratePrompt(text_client)
    final_prompt = await text_generator.generate_prompt(prompt)
    print(final_prompt)
    size = "1024x1024"  # Default size, or dynamically calculated if you prefer

    # Initialize generator and generate images
    generator = generation.GenerateImage(image_client, "dall-e-3", final_prompt, size, "standard", "natural", 1)
    arts = await generate_images(generator, req.n)
    arts = list(arts)

    # Return images in ZIP file
    zip_buffer = create_zip_from_images(arts)
    now = datetime.now()

    # Print the time in HH:MM:SS format
    print("Current Time:", now.strftime("%H:%M:%S"))
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": "attachment; filename=generated_art.zip"
    })
