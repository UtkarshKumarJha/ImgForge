from PIL import Image
import os

# Shows image dimensions so we can calculate correct bbox
for f in os.listdir("data/id_samples"):
    img = Image.open(f"data/id_samples/{f}")
    print(f"{f}: {img.size}  (width x height)")