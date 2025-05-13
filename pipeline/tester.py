import requests

url = "http://127.0.0.1:8000/custom-on-wall/"

files = [
    ("art_images", open("C:/Users/ali/Downloads/art1.jpg", "rb")),
    ("art_images", open("C:/Users/ali/Downloads/art2.jpg", "rb")),
]

# Send `box` as multiple values
data = [
    ("api_key", "samplekey"),
    ("wall_image", "https://media.istockphoto.com/id/1933752815/photo/modern-interior-of-living-room-with-leather-armchair-on-wood-flooring-and-dark-blue-wall.jpg?s=612x612&w=0&k=20&c=KqVE2Sh7Mjx_EBQC3bN1X3YPyCtcMCttKKB0aKnFN3E="),
    ("box", "0.2"),
    ("box", "0.3"),
    ("box", "0.6"),
    ("box", "0.7")
]

response = requests.post(url, data=data, files=files)

# Save the result
with open("output_response.multipart", "wb") as f:
    f.write(response.content)

# Debug output
print("Status code:", response.status_code)
print("Response headers:", response.headers)
print("First part of response:", response.content[:200])
