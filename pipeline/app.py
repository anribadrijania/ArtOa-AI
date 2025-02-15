from fastapi import FastAPI, HTTPException
from ultralytics import YOLO
from typing import List
import generation
import segmentation
import utils
import asyncio

app = FastAPI()
model = YOLO("segmentation-v1.pt")


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


@app.post("/generate-images/")
async def main(image_url: str, prompt: str, tags: List[str], box: List[float], n: int = 4):
    try:
        wall = await utils.fetch_image(image_url)
        if wall is None:
            raise HTTPException(status_code=400, detail="Invalid image URL or image could not be fetched.")
        wall.show()
        box_width, box_height = utils.get_box_coordinates(wall, box)[:2]
        size = utils.get_best_size(box_width, box_height)
        gen_model, quality = "dall-e-3", "standard"
        prompt = prompt + ", " + ", ".join(tags)

        segmentor = segmentation.Segment(model)
        generator = generation.Generate(gen_model, prompt, size, quality, 1)

        (masks, combined_masks, cropped_objects), generated_images = await asyncio.gather(
            segment_image(segmentor, wall),
            generate_images(generator, n)
        )

        final_images = []
        if masks is None:
            for art in generated_images:
                wall_art = utils.place_art_in_box(wall, art, box)
                final_images.append(wall_art)
            return final_images

        for art in generated_images:
            wall_art = utils.place_art_in_box(wall, art, box)
            final_image = utils.return_cropped_object(wall_art, cropped_objects, combined_masks)
            final_images.append(final_image)
        return final_images
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

