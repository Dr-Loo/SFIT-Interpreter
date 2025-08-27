import numpy as np
from PIL import Image
import pandas as pd
import cv2

# Load the image
image_path = r"C:\Users\Herminia Puodzius\Desktop\Provisional Patent Application\Fragment Ohi_lock Domains Sthoch_PDE.png"
img = Image.open(image_path)
img_array = np.array(img)  # Shape: (height, width, 3) for RGB

# Flatten coordinates and RGB values
height, width, _ = img_array.shape
x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
data = pd.DataFrame({
    "x": x_coords.flatten(),
    "y": y_coords.flatten(),
    "R": img_array[:, :, 0].flatten(),
    "G": img_array[:, :, 1].flatten(),
    "B": img_array[:, :, 2].flatten()
})

# Save to CSV
data.to_csv(r"C:\Users\Herminia Puodzius\Desktop\domain_data.csv", index=False)
print("CSV saved to your desktop!")