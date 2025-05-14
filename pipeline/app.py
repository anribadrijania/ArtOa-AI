"""
Description: This file contains the FastAPI application that serves as the main entry point for the pipeline.
It includes endpoints to generate AI art or upload custom art and place it on a wall image.
The pipeline uses the YOLO model for segmentation and the OpenAI DALL-E model for image generation.
"""

# Import required libraries
from fastapi import FastAPI, HTTPException, UploadFile, Request, Form, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from logger import log_debug, log_info, log_warning, log_error
from transformers import AutoModelForImageSegmentation
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from openai import AsyncOpenAI
from dotenv import load_dotenv
from PIL import Image
from typing import List, Optional
from pydantic import BaseModel
import numpy as np
import os
import torch
import time
import utils
import segmentation
import generation
import asyncio
import traceback
import io

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_API_KEY = os.getenv("APP_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set!")
if not APP_API_KEY:
    raise ValueError("APP_API_KEY is not set!")

# Create FastAPI app instance
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define device
device = "cpu"
log_debug(f"App is running on {device}...")

# Load segmentation models
remover_model = AutoModelForImageSegmentation.from_pretrained("./pretrained", trust_remote_code=True)
remover_model.load_state_dict(torch.load("remover_v2.pth", map_location=torch.device('cpu')))
remover_model.eval()
log_info("Remover model loaded.")

rcnn_model = maskrcnn_resnet50_fpn_v2()
rcnn_model.load_state_dict(torch.load("./maskrcnn_v2.pth", map_location=torch.device('cpu')))
rcnn_model.eval()
log_info("MaskRCNN model loaded.")

# OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Request schema
class GenerateRequest(BaseModel):
    api_key: str
    image_url: str
    prompt: str
    box: List[float]
    tags: List[str] = [""]
    n: int = 4

# Validate requests
def validate_request(api_key, box):
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

# Error handling middleware
@app.middleware("http")
async def add_logging_and_error_handling(request: Request, call_next):
    try:
        log_info(f"{request.method} {request.url}")
        response = await call_next(request)
        return response
    except HTTPException as e:
        log_error(f"HTTPException: {e.detail}")
        raise
    except Exception as e:
        tb = traceback.format_exc()
        log_error(f"Unhandled Exception: {tb}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Helper: segmentation
async def segment_image(rcnn_segmentor, remover_segmentor, wall):
    remover_mask = await remover_segmentor.predict_masks(wall)
    final_masks = await rcnn_segmentor.predict_masks(wall, remover_mask)
    if final_masks is None or final_masks.size == 0:
        log_warning("No objects found during segmentation.")
        return None
    return final_masks

# Helper: generation
async def generate_images(generator, n):
    tasks = [utils.generate_and_fetch(generator) for _ in range(n)]
    return await asyncio.gather(*tasks)

# Helper: process art on wall
async def process_wall_and_arts(wall, arts, box, segmentors):
    box_width, box_height, x_min, y_min = utils.get_box_coordinates(wall, box)
    masks = await segment_image(*segmentors, wall)
    final_images = []
    for art in arts:
        background_np = np.array(wall)
        art_np = np.array(art)
        h, w_img = background_np.shape[:2]
        box_percent = [x_min / w_img, y_min / h, (x_min + box_width) / w_img, (y_min + box_height) / h]
        wall_art = utils.apply_lighting_and_texture(background_np, art_np, box_percent)
        final = utils.return_cropped_objects(wall_art, masks) if masks.size is not 0 else Image.fromarray(wall_art)
        final_images.append(final)
    return final_images

# Multipart response

import urllib.parse

def create_multipart_response(images: List[Image.Image], image_format="PNG", content_type="image/png", filenames: Optional[List[str]] = None, boundary="BOUNDARY") -> Response:
    multipart_body = b""
    for i, img in enumerate(images):
        filename = filenames[i] if filenames and i < len(filenames) else f"image_{i}.{image_format.lower()}"
        safe_filename = urllib.parse.quote(filename)  # RFC 5987 encoding

        img_bytes = io.BytesIO()
        img.save(img_bytes, format=image_format)
        img_bytes.seek(0)
        image_data = img_bytes.read()

        part = (
            f"--{boundary}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Disposition: inline; filename*=utf-8''{safe_filename}\r\n\r\n"
        ).encode("utf-8") + image_data + b"\r\n"

        multipart_body += part

    multipart_body += f"--{boundary}--\r\n".encode("utf-8")

    return Response(content=multipart_body, media_type=f"multipart/mixed; boundary={boundary}")


@app.post("/generate-on-wall/")
async def generate_on_wall(req: GenerateRequest):
    start = time.time()
    validate_request(req.api_key, req.box)
    wall = await utils.fetch_image(req.image_url)
    if wall is None:
        raise HTTPException(status_code=400, detail="Invalid image URL")

    box_width, box_height, *_ = utils.get_box_coordinates(wall, req.box)
    size = utils.get_best_size(box_width, box_height)
    prompt = utils.prompt_engineering(req.prompt, req.tags)

    generator = generation.Generate(client, "dall-e-3", prompt, size, "standard", "vivid", 1)
    segmentors = (
        segmentation.MaskRCNN(rcnn_model, device),
        segmentation.BgRemover(remover_model, device)
    )
    arts = await generate_images(generator, req.n)
    final_images = await process_wall_and_arts(wall, arts, req.box, segmentors)

    return create_multipart_response(final_images, filenames=[f"generated_{i}.png" for i in range(len(final_images))])

@app.post("/custom-on-wall/")
async def custom_on_wall(
    api_key: str = Form(...),
    wall_image: str = Form(...),
    box: List[float] = Form(...),
    art_images: List[UploadFile] = File(...)
):
    validate_request(api_key, box)
    wall = await utils.fetch_image(wall_image)
    if wall is None:
        raise HTTPException(status_code=400, detail="Invalid image URL")
    arts = [Image.open(file.file).convert("RGB") for file in art_images]
    segmentors = (
        segmentation.MaskRCNN(rcnn_model, device),
        segmentation.BgRemover(remover_model, device)
    )
    final_images = await process_wall_and_arts(wall, arts, box, segmentors)
    return create_multipart_response(final_images, filenames=[f"custom_{i}.png" for i in range(len(final_images))])
