from PIL import Image
import numpy as np
from glob import glob
import os

def change_color(im, c1=(222,222,222,0), c2=(255,255,255,0), tol=0):
    """
    changes solid colors to another solid color
    tol allows a range around the original color to change
    c2 by default removes the color and leaves transparency
    """
    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability
    c1_areas = (((red >= c1[0]-(c1[0]*tol)) & (red <= c1[0]+(c1[0]*tol)))
                & ((green >= c1[1]-(c1[1]*tol)) & (green <= c1[1]+(c1[1]*tol))) 
                & ((blue >= c1[2]-(c1[2]*tol)) & (blue <= c1[2]+(c1[2]*tol)))) 
    data[c1_areas.T] = c2 # Transpose back needed
    return(Image.fromarray(data))                               

def change_color_dir(wd, out_wd, c1=(222,222,222,0), c2=(255,255,255,0), tol=0):
    im_names = sorted([os.path.basename(x) for x in glob("%s/*.png" %(wd))])
    for i in im_names:
        im = Image.open("%s/%s"%(wd,i)).convert('RGBA')
        out = change_color(im, c1=c1, c2=c2, tol=tol)
        out.save("%s/%s"%(out_wd,i))

def composite_im(im, bim, out, c=[230,228,224,0], tol=0.015, alpha=0.3):
    im = Image.open(im).convert('RGBA')
    bim = Image.open(bim).convert('RGBA')
    arr = np.array(np.asarray(im))
    red,green,blue,a = np.rollaxis(arr,axis=-1)
    mask = (((red >= c[0]-(c[0]*tol)) & (red <= c[0]+(c[0]*tol))) \
            & ((green >= c[1]-(c[1]*tol)) & (green <= c[1]+(c[1]*tol))) \
            & ((blue >= c[2]-(c[2]*tol)) & (blue <= c[2]+(c[2]*tol))))
    arr[mask,3]=0 #make target color transparent
    arr[~mask,3]=255*(1-alpha) #make other colors semi-transparent
    img_mask=Image.fromarray(arr)
    cim = Image.composite(im, bim, img_mask)
    cim.save(out)

def composite_imgs(bk_wd, ov_wd, out_wd, alpha=0.3):
    """ 
    Function to join vector and satellite images 
    
    Parameters 
    ----------
    bk_wd : Working directory of satellite data.

    ov_wd : Working directory of vector data.

    out_wd : Directory for overlay data to be stored.

    """
    
    im_names = sorted([os.path.basename(x) for x in glob("%s/*.png" %(bk_wd))])
    for i in im_names:
        im = "%s/%s"%(ov_wd,i)
        bim = "%s/%s"%(bk_wd,i)
        out = "%s/%s"%(out_wd, i)
        composite_im(im, bim, out, alpha=alpha)

def image_filter(wd, c, tol):
    """ 
    Function to delete images with a faulty color.

    Parameters 
    ----------
    wd : Working directory of data of string type

    c : List of color in RGBA format

    tol : Float value between 0-1 designating range of color find
    
    """ 

    sub_srcs = ['train', 'test', 'val']
    files = glob(os.path.join('%s/%s/b'%(wd,sub_srcs[0]), '*.png'))
    for i in range(len(sub_srcs)):
        for im_name in files:
                im_name = os.path.basename(im_name)
                filename_a = '%s/%s/a/%s'%(wd,sub_srcs[i],im_name)
                filename_b = '%s/%s/b/%s'%(wd,sub_srcs[i],im_name)
                try:
                    im = Image.open(filename_b)
                except:
                    continue
                arr = np.array(np.asarray(im))
                red,green,blue,a = np.rollaxis(arr,axis=-1)
                mask = (((red >= c[0]-(c[0]*tol)) & (red <= c[0]+(c[0]*tol))) \
                    & ((green >= c[1]-(c[1]*tol)) & (green <= c[1]+(c[1]*tol))) \
                    & ((blue >= c[2]-(c[2]*tol)) & (blue <= c[2]+(c[2]*tol))))
                #remove the file from both a and b
                if(np.any(mask)):
                    os.remove(filename_a)
                    os.remove(filename_b)

def file_matcher(src_d, to_match):
    """ 
    Function to match directory files 

    Parameters 
    ----------
    src_d : Working directory of data with the desired file structure 
    
    to_match: directory of data to be modified

    """
    src_list = glob(os.path.join(src_d, '*.png'))
    src_list = np.array([os.path.basename(i) for i in src_list])
    
    # write a function call file matcher...
    change_list = glob(os.path.join(to_match, '*.png'))
    change_list = np.array([os.path.basename(i) for i in change_list])
    mask = [im not in src_list for im in change_list]
    for img in change_list[mask]:
        rm_file = '%s/%s'%(to_match,img)
        os.remove(rm_file)
    

def filter_image_match(dwd='./data/mapbox/tiles'):
    """ 
    Function to match training images with the overlay directory 
    images after manual quality control. 

    Parameters 
    ----------
    dwd : Working directory of data of string type
    """ 

    sub_srcs = ['test', 'val', 'train']
    #sub_srcs = ['test']
    for f in range(len(sub_srcs)):
        wd = '%s/%s/'%(dwd, sub_srcs[f])
        o_wd = '%s/overlays'%(wd)
        a_wd = '%s/a'%(wd) 
        b_wd = '%s/b'%(wd)
        ab_wd = '%s/ab'%(wd)
        file_matcher(o_wd, a_wd)
        file_matcher(o_wd, b_wd)
        file_matcher(o_wd, ab_wd)
