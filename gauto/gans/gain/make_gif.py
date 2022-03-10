from PIL import Image
import glob
import re

def make_gif(directory_input, directory_output):
    # Create the frames
    frames = []
    imgs = glob.glob(directory_input)
    imgs.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    
    # Save into a GIF file that loops forever
    frames[0].save(directory_output, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=300, loop=0)