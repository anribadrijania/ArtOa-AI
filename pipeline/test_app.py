from unittest.mock import AsyncMock, patch
from PIL import Image
from httpx import AsyncClient, ASGITransport
from app import app
import pytest
from io import BytesIO

transport = ASGITransport(app=app)

# Mock function to simulate OpenAI image generation response
def mock_generate_and_fetch(*args, **kwargs):
    # Simulating a generated image (dummy 100x100 blue image)
    image = Image.new('RGB', (100, 100), color=(73, 109, 137))
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()  # Return raw image data

# Mock function to simulate the hashing of the API key
def mock_hash_api_key(api_key: str):
    # Mocked behavior: simulate that the provided API key is valid
    if api_key == "valid_api_key":
        return "valid_key"  # Return the expected valid hashed key
    else:
        return "invalid_key"  # Simulate an invalid hashed key

@pytest.mark.asyncio
async def test_missing_api_key():
    # Missing API key in the request
    request_data = {
        "image_url": "https://png.pngtree.com/background/20210706/original/pngtree-hd-brick-wall-background-picture-image_206286.jpg",
        "prompt": "abstract painting of a sunset",
        "box": [0.1, 0.1, 0.5, 0.5],
        "tags": ["modern", "colorful"],
        "n": 2
    }

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/generate-on-wall/", json=request_data)

    assert response.status_code == 401
    assert response.json()["detail"] == "API key is required."

@pytest.mark.asyncio
@patch("app.utils.hash_api_key", side_effect=mock_hash_api_key)
async def test_invalid_api_key(mock_hash):
    # Invalid API key in the request
    request_data = {
        "api_key": "valid_api_key",
        "image_url": "https://png.pngtree.com/background/20210706/original/pngtree-hd-brick-wall-background-picture-image_206286.jpg",
        "prompt": "abstract painting of a sunset",
        "box": [0.1, 0.1, 0.5, 0.5],
        "tags": ["modern", "colorful"],
        "n": 2
    }

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/generate-on-wall/", json=request_data)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key."

@pytest.mark.asyncio
@patch("utils.fetch_image", return_value=None)  # Simulate fetch_image failure
async def test_invalid_image_url(mock_fetch):
    request_data = {
        "api_key": "valid_api_key",
        "image_url": "https://invalid-url.com/sample.jpg",
        "prompt": "abstract painting of a sunset",
        "box": [0.1, 0.1, 0.5, 0.5],
        "tags": ["modern", "colorful"],
        "n": 2
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/generate-on-wall/", json=request_data)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image URL or image could not be fetched."

@pytest.mark.asyncio
@patch("utils.generate_and_fetch", return_value=Image.new("RGB", (256, 256), "blue"))
async def test_multiple_image_generation():
    request_data = {
        "api_key": "valid_api_key",
        "image_url": "https://png.pngtree.com/background/20210706/original/pngtree-hd-brick-wall-background-picture-image_206286.jpg",
        "prompt": "abstract painting of a sunset",
        "box": [0.1, 0.1, 0.5, 0.5],
        "tags": ["modern", "colorful"],
        "n": 3  # Requesting 3 images
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/generate-on-wall/", json=request_data)

    assert response.status_code == 200
    assert len(response.content) > 0  # Check if response contains image data
    assert len(response.files) == 3  # Check if 3 images are returned

@pytest.mark.asyncio
@patch("segmentation.MaskRCNN.predict_masks", return_value=["fake_mask"])
@patch("segmentation.BgRemover.predict_masks", return_value="fake_remover_mask")
@patch("generation.Generate.generate", return_value=Image.new("RGB", (256, 256), "blue"))
async def test_segmentation_masks(mock_generate, mock_remover, mock_rcnn):
    request_data = {
        "api_key": "valid_api_key",
        "image_url": "https://example.com/sample.jpg",
        "prompt": "abstract painting of a sunset",
        "box": [0.1, 0.1, 0.5, 0.5],
        "tags": ["modern", "colorful"],
        "n": 1
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/generate-on-wall/", json=request_data)

    assert response.status_code == 200
    assert len(response.files) == 1  # One image returned
    mock_rcnn.assert_called_once()  # Ensure the segmentation model was called
    mock_remover.assert_called_once()  # Ensure background remover was called

@pytest.mark.asyncio
@patch("generation.Generate.generate", side_effect=Exception("Internal error"))
async def test_internal_server_error(mock_generate):
    request_data = {
        "api_key": "valid_api_key",
        "image_url": "https://example.com/sample.jpg",
        "prompt": "abstract painting of a sunset",
        "box": [0.1, 0.1, 0.5, 0.5],
        "tags": ["modern", "colorful"],
        "n": 2
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/generate-on-wall/", json=request_data)

    assert response.status_code == 500
    assert "Internal server error" in response.json()["detail"]
