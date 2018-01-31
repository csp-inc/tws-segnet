import sys
sys.path.append('/contents')
from src.image_tools import change_color_dir
from src.image_tools import composite_imgs
import os

input_dir = "/contents/images/train"
a_dir = "%s/aa"%input_dir
b_dir = "%s/b"%input_dir
bb_dir = "%s/bb"%input_dir
ab_dir = "%s/ab"%input_dir
if not os.path.exists(ab_dir):
    os.makedirs(ab_dir)

#need to change color first
change_color_dir(b_dir, bb_dir, c1=(0,0,0,0), c2=(255,0,0,0))
composite_imgs(a_dir, bb_dir, ab_dir, 0.65)
