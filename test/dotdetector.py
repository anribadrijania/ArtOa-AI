import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

def show_image_and_click_for_coordinates(image_url):
    # Load image from URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    height, width, _ = image_np.shape
    window_name = 'Click to get % coords'

    display_img = image_np.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal display_img
        if event == cv2.EVENT_LBUTTONDOWN:
            x_percent = x / width
            y_percent = y / height
            display_img = image_np.copy()  # Reset to original
            text = f"x_min: {x_percent:.4f}, y_min: {y_percent:.4f}"
            cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, display_img)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.waitKey(10)

    cv2.destroyAllWindows()



image_url = input("url: ")
show_image_and_click_for_coordinates(image_url)