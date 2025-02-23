from fastapi.testclient import TestClient
import app  # Import the FastAPI app

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
        "box": [0.1, 0.15, 0.35, 0.4],
        "n": 3  # Number of art variations
    })

    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert isinstance(response.json(), list), "Response is not a list"
