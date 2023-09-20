from PIL import Image
import glob
import os

# Create the frames
frames = []
imgs = sorted(glob.glob("*.png"), key=os.path.getmtime)

for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save(
    "png_to_gif.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=1,
    loop=0,
)
