import cv2
import numpy as np
import aiohttp
import hashlib
from io import BytesIO
from PIL import Image
from torchvision import transforms


async def fetch_image(url):
    """
    Fetch an image from a given URL asynchronously.

    :param url: The URL of the image.
    :return: PIL Image object if successful.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                return Image.open(BytesIO(data))


async def generate_and_fetch(generator):
    """
    Generate an image using the generator class and fetch the resulting image.

    :param generator: An instance of the Generate class.
    :return: PIL Image object.
    """
    prompt, custom_image = await generator.generate_image_with_revised_prompt()
    image = await fetch_image(custom_image)
    print("prompt: ", prompt)
    return image


def get_best_size(width: int, height: int) -> str:
    """
    Determine the best image size based on aspect ratio.

    :param width: Original width of the image.
    :param height: Original height of the image.
    :return: Best matching size as a string.
    """
    sizes = [(1024, 1024), (1792, 1024), (1024, 1792)]
    box_ratio = width / height
    best_size = min(sizes, key=lambda size: abs(box_ratio - size[0] / size[1]))

    return f"{best_size[0]}x{best_size[1]}"


def combine_masks(image, masks):
    """
    Merge multiple masks into a single refined mask.

    :param image: The original image.
    :param masks: A list of masks to combine.
    :return: The combined mask.
    """
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
    """
    Crop an object from an image using a given mask.

    :param image: The input image.
    :param mask: The mask defining the object to be cropped.
    :return: Cropped object image.
    """
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
    Paste a cropped object back onto the original image.

    :param image: The base image.
    :param cropped_objects: The cropped object.
    :param mask: The mask defining the object's position.
    :return: Modified image with the object pasted back.
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
    """
    Convert percentage-based bounding box coordinates into pixel values.

    :param wall: The base image.
    :param box: The bounding box in percentage values.
    :return: Pixel-based bounding box coordinates.
    """
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


def place_art_in_box(wall, art, box_width, box_height, x_min, y_min):
    """
    Place an art image inside a defined bounding box on a wall image.

    :return: The modified wall image with the art placed inside the box.
    """
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

    resized_art = art.resize((new_width, new_height))

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


def preprocess_image(image, device):
    image_size = (1024, 1024)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if device == "cuda":
        return transform(image).unsqueeze(0).to("cuda").half()
    else:
        return transform(image).unsqueeze(0).cpu()


def return_cropped_objects(image, masks):
    """
    Overlay a transparent image of cropped objects onto the background image.

    :param image: The base image (background).
    :param cropped_objects: The transparent image containing cropped objects.
    :return: Modified image with objects pasted back.
    """
    try:
        cropped_objects = Image.fromarray(masks)
        image = image.convert("RGBA")
        cropped_objects = cropped_objects.convert("RGBA")

        result_image = Image.alpha_composite(image, cropped_objects)
        return result_image
    except Exception as e:
        print(f"Error: {e}")
        return None


def transformer_for_rcnn(image, device):
    image_np = np.array(image)
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensors = transform(image).unsqueeze(0).to(device)

    return image_np, input_tensors


def apply_lighting_and_texture(background: np.ndarray, artwork: np.ndarray, box_percent: list) -> Image.Image:
    h, w, _ = background.shape
    x_min_px = int(box_percent[0] * w)
    y_min_px = int(box_percent[1] * h)
    x_max_px = int(box_percent[2] * w)
    y_max_px = int(box_percent[3] * h)

    box_region = background[y_min_px:y_max_px, x_min_px:x_max_px]
    box_region_float = box_region.astype(np.float32) / 255.0
    illum_color = cv2.GaussianBlur(box_region_float, (51, 51), 0)  # blurred RGB lighting

    # Compute grayscale version (intensity)
    illum_gray = cv2.cvtColor(illum_color, cv2.COLOR_RGB2GRAY)[..., np.newaxis]

    # Compute color strength: how far color is from grayscale
    color_strength = np.linalg.norm(illum_color - illum_gray, axis=2, keepdims=True)  # (H, W, 1)

    # Normalize strength between 0 (gray) and 1 (highly colored)
    max_strength = 1.0  # tweak: higher = less sensitive to color
    blend_factor = np.clip(color_strength / max_strength, 0, 1)

    # Final illumination map: blend between grayscale and colored
    illum_blend = illum_gray * (1 - blend_factor) + illum_color * blend_factor

    # Optional: clamp to avoid extreme lighting
    illum_blend = np.clip(illum_blend, 0.05, 1.5)

    box_height = y_max_px - y_min_px
    box_width = x_max_px - x_min_px

    artwork_resized = cv2.resize(artwork, (box_width, box_height))
    artwork_float = artwork_resized.astype(np.float32) / 255.0

    # Apply blended lighting
    artwork_lit = artwork_float * illum_blend
    artwork_lit = np.clip(artwork_lit, 0, 1)

    # Apply subtle wall texture
    texture_scaled = cv2.resize(box_region, (box_width, box_height))
    texture_float = texture_scaled.astype(np.float32) / 255.0
    texture_overlay = texture_float * 0.05 + artwork_lit * 0.95
    texture_overlay = np.clip(texture_overlay, 0, 1)

    # Edge fade mask
    alpha_channel = np.ones_like(texture_overlay[..., 0], dtype=np.float32)
    fade_width = int(0.01 * box_height)
    mask = np.ones((box_height, box_width), dtype=np.float32)

    x_fade = np.linspace(0, 1, fade_width)
    mask[:, :fade_width] = x_fade
    mask[:, -fade_width:] = x_fade[::-1]
    y_fade = np.linspace(0, 1, fade_width)
    mask[:fade_width, :] = y_fade[:, np.newaxis]
    mask[-fade_width:, :] = y_fade[::-1, np.newaxis]
    alpha_channel *= np.clip(mask, 0.0, 1.0)

    # Blend final result into background
    result_image = background.astype(np.float32) / 255.0
    result_image[y_min_px:y_max_px, x_min_px:x_max_px, :3] = (
        result_image[y_min_px:y_max_px, x_min_px:x_max_px, :3] * (1 - alpha_channel[..., np.newaxis]) +
        texture_overlay * alpha_channel[..., np.newaxis]
    )

    result_image_uint8 = (np.clip(result_image, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(result_image_uint8)


def hash_api_key(api_key):
    """
    Hash the API key using SHA-256.

    :param api_key: The API key to hash.
    :return: The hashed API key.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()