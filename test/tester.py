import requests
import zipfile
import io
from PIL import Image

# API endpoint
url = "http://localhost:8000/custom-on-wall/"

# Replace with your actual API key
api_key = "samplekey"

# Sample inputs
wall_image_url = "https://images.stockcake.com/public/0/2/9/02989daf-35f3-43d6-8a13-7ba5e6a95859_large/shadows-meet-light-stockcake.jpg"
art_image_urls = [
    "https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera-course-photos.s3.amazonaws.com/65/78ce0081ad11e681d7bb31b0a632ef/starry-night.jpg"
]
box = [0.0, 0.0, 1.0, .9]  # [x_min, y_min, x_max, y_max]

# Form data
data = {
    "api_key": api_key,
    "wall_image": wall_image_url,
    "box": box,  # Sending box as a JSON string (e.g., "[0.2, 0.3, 0.7, 0.8]")
    "image_urls": art_image_urls  # Sending image_urls as JSON string
}

# Make POST request
response = requests.post(url, json=data)

# Handle response
if response.status_code == 200:
    zip_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_bytes) as zip_file:
        for filename in zip_file.namelist():
            with zip_file.open(filename) as img_file:
                img = Image.open(img_file)
                img.show()
else:
    print("Error:", response.status_code, response.text)

"""
{
  "api_key": "samplekey",
  "wall_image": "https://images.photowall.com/interiors/60421/landscape/wallpaper/room88.jpg?w=2000&q=80",
  "box": [
    0.1, 0.1, 0.9, 0.9
  ],
  "image_urls": [
    "https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera-course-photos.s3.amazonaws.com/65/78ce0081ad11e681d7bb31b0a632ef/starry-night.jpg"
  ]
}
"""