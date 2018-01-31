from PIL import Image
from glob import glob

size = 600
wd = './data/windows_3600'
files = glob('%s/*.png'%(wd))
for im in files:
    a = Image.open(im)
    a = a.resize((size, size), Image.BICUBIC)
    a.save(im)
