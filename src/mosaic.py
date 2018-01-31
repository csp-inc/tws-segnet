import numpy as np
from PIL import Image
from glob import glob

img_nums = ['3711401', '3711402', '3711403']
i=0

# get the images by identifier lat and lon 
imgs = glob('./modeled/*%s*.png'%(img_nums[i]))
# need to break up the image by date...

