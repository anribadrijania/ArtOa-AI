from fastapi.testclient import TestClient
from app import app  # Import the FastAPI app

# Initialize the test client
client = TestClient(app)


def test_generate_images():
    """
    Test the /generate-on-wall/ endpoint to ensure it returns a list of generated images.
    """
    response = client.post("/generate-on-wall/", json={
        "image_url": "https://images.plusmood.com/wp-content/uploads/2024/08/painted-feature-wall.jpg",
        "prompt": "futuristic city",
        "tags": ["sci-fi", "neon"],
        "box": [0.2, 0.2, 0.7, 0.7],
        "n": 1  # Number of art variations
    })

    assert response.status_code == 200
    data = response.json()

    assert "images" in data, "Response should contain 'images' key"
    assert isinstance(data["images"], list), "Images should be a list"
    assert len(data["images"]) == 2, f"Expected 2 images, got {len(data['images'])}"
    assert all(img.startswith("/static/") for img in data["images"]), "Invalid image URL format"

    print("Test passed: Images are generated and returned correctly")
