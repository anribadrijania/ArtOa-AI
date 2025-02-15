import cv2
import numpy as np
import aiohttp
import asyncio
from io import BytesIO
from PIL import Image


async def fetch_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                return Image.open(BytesIO(data))


async def generate_and_fetch(generator):
    custom_image = await generator.generate_image()
    await fetch_image(custom_image)


def get_best_size(width: int, height: int) -> str:
    sizes = [(1024, 1024), (1792, 1024), (1024, 1792)]
    box_ratio = width / height
    best_size = min(sizes, key=lambda size: abs(box_ratio - size[0] / size[1]))

    return f"{best_size[0]}x{best_size[1]}"


def combine_masks(image, masks):
    # Combine masks into a single mask
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    for mask in masks:
        eroded_mask = cv2.erode(mask, kernel, iterations=2)
        refined_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
        combined_mask = np.maximum(combined_mask, (refined_mask * 255).astype(np.uint8))

    # Resize the mask back to the original image size
    combined_mask = cv2.resize(combined_mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
    return combined_mask


def crop_object_with_mask(image, mask):
    try:
        image = image.convert("RGBA")
        mask = Image.fromarray(mask).convert("L")

        transparent_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        transparent_image.paste(image, mask=mask)

        bbox = mask.getbbox()
        cropped_image = transparent_image.crop(bbox)

        return cropped_image
    except Exception as e:
        print(f"Error: {e}")
        return None


def return_cropped_object(image, cropped_objects, mask):
    """
    Convert the original image to black and white before pasting the cropped object back.
    """
    try:
        image = image.convert("RGBA")
        cropped_objects = cropped_objects.convert("RGBA")

        mask = Image.fromarray(mask).convert("L")

        result_image = image.copy()

        bbox = mask.getbbox()
        result_image.paste(cropped_objects, bbox, mask=cropped_objects)

        return result_image
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_box_coordinates(wall, box):
    # Get base image dimensions
    base_width, base_height = wall.size

    # Convert box coordinates from percentages to pixel values
    x_min = int(box[0] * base_width)
    y_min = int(box[1] * base_height)
    x_max = int(box[2] * base_width)
    y_max = int(box[3] * base_height)

    # Calculate box width and height
    box_width = x_max - x_min
    box_height = y_max - y_min

    return box_width, box_height, x_min, y_min


def place_art_in_box(wall, art, box):
    box_width, box_height, x_min, y_min = get_box_coordinates(wall, box)

    # Resize art while maintaining aspect ratio
    art_aspect_ratio = art.width / art.height
    box_aspect_ratio = box_width / box_height

    if art_aspect_ratio > box_aspect_ratio:
        # Match width and scale height
        new_width = box_width
        new_height = int(box_width / art_aspect_ratio)
    else:
        # Match height and scale width
        new_height = box_height
        new_width = int(box_height * art_aspect_ratio)

    resized_art = art.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a blank canvas the size of the box with a transparent background
    canvas = Image.new("RGBA", (box_width, box_height), (255, 255, 255, 0))

    # Calculate position to center the resized art within the box
    x_offset = (box_width - new_width) // 2
    y_offset = (box_height - new_height) // 2

    # Paste the resized art onto the canvas
    canvas.paste(resized_art, (x_offset, y_offset), resized_art if resized_art.mode == 'RGBA' else None)

    # Paste the canvas onto the base image
    base_image = wall.convert("RGBA")
    base_image.paste(canvas, (x_min, y_min), canvas)

    return base_image
