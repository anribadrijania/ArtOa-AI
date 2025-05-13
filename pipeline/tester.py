import requests

url = "http://127.0.0.1:8000/custom-on-wall/"

files = [
    ("wall_image", open("C:/Users/alili/Downloads/wall.jpg", "rb")),
    ("art_images", open("C:/Users/alili/Downloads/art1.jpg", "rb")),
    ("art_images", open("C:/Users/alili/Downloads/art2.jpg", "rb")),
]

# Send `box` as multiple values
data = [
    ("api_key", "f2ece78de8e08f6abb5612199455dd60401afe20a5f9790c1a3803ff466d1887"),
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
