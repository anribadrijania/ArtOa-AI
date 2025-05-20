import requests

url = "http://127.0.0.1:8000/custom-on-wall/"

files = [
    ("art_images", open("C:/Users/alili/Downloads/art1.jpg", "rb")),
    ("art_images", open("C:/Users/alili/Downloads/art2.jpg", "rb")),
]

# Send `box` as multiple values
data = [
    ("api_key", "samplekey"),
    ("wall_image", "https://thumbs.dreamstime.com/b/pink-wall-sunlight-shadows-bright-interior-space-modern-design-minimalistic-aesthetic-ai-generated-356586077.jpg"),
    ("box", "0.2"),
    ("box", "0.1"),
    ("box", "0.9"),
    ("box", "0.75")
]

response = requests.post(url, data=data, files=files)

# Save the result
with open("output_response.multipart", "wb") as f:
    f.write(response.content)

# Debug output
print("Status code:", response.status_code)
print("Response headers:", response.headers)
print("First part of response:", response.content[:200])
