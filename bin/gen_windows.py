import strider 
from glob import glob

img_files = glob('*.tif')
imgs = [strider.Image(i) for i in img_files]

# here we need to integrate the SegNet model to predict on our 
# generated windows.

