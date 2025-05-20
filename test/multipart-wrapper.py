from requests_toolbelt.multipart.decoder import MultipartDecoder
from PIL import Image
import io

# Load multipart content from file
with open("./output_response.multipart", "rb") as f:
    data = f.read()

# Specify the boundary you used
boundary = "BOUNDARY"
content_type = f"multipart/mixed; boundary={boundary}"

# Decode the multipart content
decoder = MultipartDecoder(data, content_type)

# Extract and open images
images = []
for part in decoder.parts:
    content_type = part.headers.get(b"Content-Type", b"").decode()
    if content_type.startswith("image/"):
        image = Image.open(io.BytesIO(part.content))
        images.append(image)
        image.show()  # Optional
