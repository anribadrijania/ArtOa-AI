import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tkinter import filedialog
from tkinter import Tk
from PIL import Image

# Use Tk to open a file dialog
root = Tk()
root.withdraw()  # Hide the main window
file_path = filedialog.askopenfilename(title="Select an Image File")

# Load image using PIL
img = Image.open(file_path)
img_width, img_height = img.size

# Convert to something matplotlib can display
img_array = mpimg.imread(file_path)

# Display image
fig, ax = plt.subplots()
ax.imshow(img_array)
ax.set_title("Click a point on the image")

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)

        # Calculate percentage
        x_pct = (x / img_width) * 100
        y_pct = (y / img_height) * 100

        print(f"Clicked pixel: ({x}, {y})")
        print(f"Position in percentages: ({x_pct:.2f}%, {y_pct:.2f}%)")

# Connect click event
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
