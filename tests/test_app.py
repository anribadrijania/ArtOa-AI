from fastapi.testclient import TestClient
from pipeline import app

client = TestClient(app)


def test_generate_images():
    response = client.post("/generate-on-wall/", json={
        "image_url": "",  # the image of the wall. example: "https://example.com/sample-wall.jpg"
        "prompt": "",     # example: "futuristic city"
        "tags": [],       # example: ["sci-fi", "neon"]
        "box": [],        # box coordinates. example: [100, 150, 350, 400]
        "n": 4            # number of art variations. min=1
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# Needs to be updated ...
