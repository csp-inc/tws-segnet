import numpy as np
from src.image_tools import change_color_dir,\
        file_matcher, composite_imgs
from PIL import Image
from glob import glob
import gdal
import os

def convert_img_type(wd, fromtype='jpg', totype='png', rmfile=True):
    '''
    Converts image file type within a diretory

    Parameters:
    -----------
    wd : <str> directory where image files are located
    fromtype : <str> from extension type
    totype : <str> to extension type
    rmfile : <bool> flag to remove original file 
    '''
    imgs = sorted(glob('%s/*.%s'%(wd, fromtype)))
    for im in imgs:
        outfile = '%s.%s'%(os.path.splitext(os.path.basename(im))[0],totype)
        img = Image.open(im)
        img.save('%s/%s'%(wd,outfile))
        if rmfile:
            os.remove(im)
    return()

def move_data(data_list, outdir):
    '''
    Input a list of images in order
    <x_datafilename> <y_datafilename>
    as a .txt file. 
    Will convert the image files to RGB 
    (removes Alpha channel), and changes
    to Numpy array type, and move all 
    corresponding images to local
    project outdir.
    
    Parameters:
    -----------
    data_list : str of text file of x and y images
    '''

    F = open(data_list,'r')
    x_raw = []
    y_raw = []

    os.makedirs('%s/a'%(outdir), exist_ok=True)
    os.makedirs('%s/b'%(outdir), exist_ok=True)
    for l in F:
        x = l.rstrip().split(" ",)[0]
        y = l.rstrip().split(" ",)[1]
        a = Image.open(x).convert('RGB')
        b = Image.open(y).convert('RGB')
        a.save('%s/a/%s'%(outdir,os.path.basename(x)))
        b.save('%s/b/%s'%(outdir,os.path.basename(y)))
        #a.save('/contents/images/train/a/%s'%os.path.basename(x))
        #b.save('/contents/images/train/b/%s'%os.path.basename(y))
    return()

def make_overlays(img_dirc="/contents/images/train"): 
    a_dir = "%s/a"%img_dirc
    b_dir = "%s/b"%img_dirc
    bb_dir = "%s/bb"%img_dirc
    ab_dir = "%s/ab"%img_dirc
    if not os.path.exists(ab_dir):
        os.makedirs(ab_dir)
    if not os.path.exists(bb_dir):
        os.makedirs(bb_dir)

    #need to change color first
    change_color_dir(b_dir, bb_dir, c1=(0,0,0,0), c2=(255,0,0,0))
    composite_imgs(a_dir, bb_dir, ab_dir, 0.65)
    return()

def match_data_dircs(input_dir="./images/train"): 
    # for the a files 
    src_d = "%s/ab"%input_dir 
    to_match = "%s/a"%input_dir 
    file_matcher(src_d, to_match) 
    # for the b files 
    to_match = "%s/b"%input_dir 
    file_matcher(src_d, to_match) 
    return()

def get_data(img_dirc):
    '''
    Input directory of images 
    <x_datafilename> <y_datafilename>
    Will convert the image files to RGB 
    (removes Alpha channel), and changes
    to Numpy array type.
    
    Parameters:
    -----------
    data_list : str of text file of x and y images
    '''
    img_files = sorted(glob('%s/*.png'%img_dirc))
    im_raw = []
    for l in range(len(img_files)):
        im = img_files[l]
        a = Image.open(im).convert('RGB')
        im_raw.append(np.array(a))
    return(np.array(im_raw))

def get_xy_data(img_dirc, batch_size=None):
    '''
    Input directory of images 
    x_datafilename <str>,
    y_datafilename <str>,
    Will convert the image files to RGB 
    (removes Alpha channel), and changes
    to Numpy array type.
    
    Parameters:
    -----------
    data_list : str of text file of x and y images
    '''
    x_dirc = '%s/a'%img_dirc
    y_dirc = '%s/b'%img_dirc
    x_files = sorted(glob('%s/*.png'%x_dirc))
    y_files = sorted(glob('%s/*.png'%y_dirc))
    x_raw = []
    y_raw = []
    batch = np.arange(len(x_files))
    if batch_size is not None:
        batch = np.random.choice(batch, batch_size, replace=False)
    for l in batch:
        x = x_files[l]
        y = y_files[l]
        a = Image.open(x).convert('RGB')
        b = Image.open(y).convert('RGB')
        x_raw.append(np.array(a))
        y_raw.append(np.array(b))
    return(np.array(x_raw), np.array(y_raw))

def data_augmentation(x, y, multiple=1):
    '''
    Takes in pre-processed x and y data, 
    and randomly transforms it with a 
    rotation, flip, or mirror. 
    Concatenates new data to end of inputs

    Parameters:
    x : numpy array of raw 4D x training data
    y : numpy array of raw 4D y training data
    multiple : factor of how many additional permuted
    images desired (max = 1).
    '''
    seed = [np.random.choice([-1,0,1]), \
            np.random.choice([0,1,2])]
    # create a seed for rotation scale -1, 0, or 1
    # flip/mirror/neither : 0-2,
    # apply these values to the new data generated
    im_x = Image.fromarray(x).rotate(90*seed[0])
    im_y = Image.fromarray(y).rotate(90*seed[0])
    if seed[1] == 0: #flip
        im_x = im_x.transpose(Image.FLIP_LEFT_RIGHT)
        im_y = im_y.transpose(Image.FLIP_LEFT_RIGHT)
    if seed[1] == 1: #mirror
        im_x = im_x.transpose(Image.FLIP_TOP_BOTTOM)
        im_y = im_y.transpose(Image.FLIP_TOP_BOTTOM)
    x_ = np.array(im_x)
    y_ = np.array(im_y)
    return(x_,y_)

def pre_process_x(a):
    '''
    Takes a raw array of images, add_channels the data, converts to float, and normalizes
    '''
    #out = np.expand_dims((a.astype('float16')/255.), axis=0)
    out = (a.astype('float16')/255.)
    return(out)

def pre_process_y(y, n_classes):
    '''
    Takes a raw array of labeled data for semantic segmentation
    Assumes that all channels are identical and needs array to be 
    collapsed to a one channel image and a one-hot categorization
    applied.
    
    Parameters:
    -----------
    y : 3d numpy array;
        first dimension is the number of images
        second dimension is the rows
        third dimension is the cols
        fourth dimension is the number of channels
    
    num_classes : int; 
        specification of number of unique classes
    '''
    y_oneband = y[:,:,0]
    y_out = np.zeros((y.shape[0], y.shape[1], n_classes))
    for i in range(n_classes):
        y_out[:,:,i][y_oneband==i] = 1
    out = y_out.reshape((y_out.shape[0]*y_out.shape[1], n_classes))
    #out = np.expand_dims(out.astype('uint8'), axis=0)
    out = out.astype('uint8')
    return(out)

def data_generator(img_dirc, augment_img=False, batch_size=None, weights_out=False):
    '''
    Input directory of images 
    <x_datafilename> <y_datafilename>
    Will convert the image files to RGB 
    (removes Alpha channel), and changes
    to Numpy array type.
    
    Parameters:
    -----------
    data_list : str of text file of x and y images
    '''
    x_dirc = '%s/a'%img_dirc
    y_dirc = '%s/b'%img_dirc
    x_files = sorted(glob('%s/*.png'%x_dirc))
    y_files = sorted(glob('%s/*.png'%y_dirc))
    total_data_size = len(x_files)
    data_indices = np.arange(total_data_size)
    if batch_size is None:
        batch_size = total_data_size
    n_batches = total_data_size // batch_size
    batch_list = np.random.permutation(total_data_size)
    for b in range(n_batches):
        permutation = batch_list[b*batch_size:((b+1)*batch_size)]
        batch = data_indices[permutation]
        x_batch = []
        y_batch = []
        for l in batch:
            x = x_files[l]
            y = y_files[l]
            a = np.array(Image.open(x).convert('RGB'))
            b = np.array(Image.open(y).convert('RGB'))
            if augment_img:
                a,b = data_augmentation(a,b)
            x_ = pre_process_x(a)
            y_ = pre_process_y(b,len(np.unique(np.array(b))))
            x_batch.append(x_)
            y_batch.append(y_)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        if weights_out:
            yield (x_batch, y_batch, get_weights(y_batch))
        else:
            yield (x_batch, y_batch)

def get_weights(y):
    '''
    Take in a pre-processed y numpy array for input and returns Kendall et al 2015
    weighting method for class weights based on the median of frequecy
    divided by the quantity of total number of pixels for a class
    divided by the number of total pixels from images that class exists.
    
    Parameters:
    y : 3d numpy array;
        first dimension is the number of images
        second dimension is the total number of pixels in image 
        third dimension is the number of channels
    '''
    #calculate the frequency of each class type
    y_samples = y.shape[0]
    y_pix = y.shape[1]
    n_weights = y.shape[2]
    c_im_pix = np.zeros(n_weights)
    c_pix = np.zeros(n_weights)
    for c in range(n_weights):
        c_pix[c] = np.sum(y[:,:,c])
        for s in range(y_samples):
            if np.any(y[s,:,c] != 1):
                c_im_pix[c] += y_pix
    #calculate the total number of pixels from where that class is present
    freq_c = c_pix/c_im_pix
    final_class_weights = np.median(freq_c)/freq_c
    return(final_class_weights)

def normalize_stack(imgs):
    '''
    Takes in a stack of <np.array> images of
    dimensions <N,rows,cols> and outputs a
    0 to 1 normalized dataset

    Parameters:
    -----------
    imgs : numpy array of <float> or <int>
    (N,rows,cols)
    '''
    return((imgs-np.min(imgs))/(np.max(imgs)-np.min(imgs)))

def create_subdirectories_from_filenames():
    '''
    Builds subdirectories given a list of unsorted file names and string indices to search
    
    !!!DEPRECATED CODE!!!
    
    Parameters:
    ----------
    '''
    # this is a funciton to move all the data to the appropriate image directory and date
    for d in dates:
        os.makedirs('%s/%s'%(wd,d), exist_ok=True)
        date_files = [x for x in imgs if '%s'%(d) in x]   
        for i in img_nums:
            os.makedirs('%s/%s/%s'%(wd,d,i), exist_ok=True)
            img_files = [y for y in date_files if '%s'%(i) in y]   
            for q in qq:
                qq_files = [z for z in img_files if '%s'%(q) in z]
                if (len(qq_files)==0): #sometimes we don't have that quadrat
                    continue
                else:
                    os.makedirs('%s/%s/%s/%s'%(wd,d,i,q), exist_ok=True)        
                    [shutil.copy(o,'%s/%s/%s/%s'%(wd,d,i,q)) for o in qq_files if '%s'%(q) in o]
    return()

def export_tif(image, ref_tif, outname, alpha_mask=None, dtype=gdal.GDT_Byte):
    '''
    Input a numpy array image and a reference geotif 
    to convert image to geotiff of same geotransform
    and projection. Note, if alpha_mask is not None,
    creates a 4 channel geotiff (alpha as last channel)

    Parameters:
    -----------
    image - 3D <numpy array>
    ref_tif - Geotiff reference <gdal object> of same dimensions
    outname - <str> file name to output to (use .tif extension)
    dtype - <str> denoting data type for GeoTiff. Defaults to 8 bit image,
    but can use gdal.GDT_Float32
    '''

    gt = ref_tif.GetGeoTransform()
    proj = ref_tif.GetProjection()
    xsize = ref_tif.RasterXSize
    ysize = ref_tif.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    if alpha_mask is None:
        bands = 1
    else:
        bands = 4
    out = driver.Create(outname, xsize, ysize, bands, dtype)
    out.SetGeoTransform(gt)
    out.SetProjection(proj)
    out.GetRasterBand(1).WriteArray(image) #if we want a red image for a 4 channel
    if bands==4: #clunky change this!
        out.GetRasterBand(2).WriteArray(np.zeros(np.shape(image))) 
        out.GetRasterBand(3).WriteArray(np.zeros(np.shape(image)))
        out.GetRasterBand(4).WriteArray(alpha_mask)
    return('created %s'%(outname))

def extend_border_mask(border_mask, bufr): 
    for n in range(np.shape(border_mask)[0]): #for the rows 
        border = np.where(np.ediff1d(border_mask[n,:]))[0]
        if len(border) == 0:
            continue
        elif len(border) == 1: #if there is one border...
            if border[0] < np.shape(border_mask)[0]/2: #left border
                border_mask[n,:border[0]+bufr] = True
            else: #right border
                border_mask[n,border[0]-bufr:] = True
        else: #2 borders
            lb = border[0]
            rb = border[1]
            border_mask[n,:lb+bufr] = True
            border_mask[n,rb-bufr:] = True

    for n in range(np.shape(border_mask)[1]): #for the columns 
        border = np.where(np.ediff1d(border_mask[:,n]))[0]
        if len(border) == 0:
            continue
        elif len(border) == 1: #if there is one border...
            if border[0] < np.shape(border_mask)[1]/2: #top border
                border_mask[:border[0]+bufr,n] = True
                border_mask[-bufr:,n] = True #seems to still have bottom border...
            else: #bottom border
                border_mask[border[0]-bufr:,n] = True
                border_mask[:bufr,n] = True #seems to still have top border...
        else:
            tb = border[0]
            bb = border[1]
            border_mask[:tb+bufr,n] = True
            border_mask[bb-bufr:,n] = True
    #seems to have some overlap still that needs to be accounted for
    border_mask[:int(bufr/2),:] = True
    border_mask[-int(bufr/2):,:] = True
    border_mask[:,:int(bufr/2)] = True
    border_mask[:,-int(bufr/2):] = True
    return(border_mask)
