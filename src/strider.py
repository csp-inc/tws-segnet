import numpy as np
import gdal
import osr
import pandas as pd
import os
import yaml
from PIL import Image as pimg

with open('/contents/src/parameters.yaml') as p_yaml:
    params = yaml.safe_load(p_yaml)
params['window_bands'] = np.array(params['window_bands'])

class Window:
    def __init__(self, parent, xindex: int, x: int, yindex: int, y: int, array: np.ndarray = None):
        self.parent = parent
        self.t_form = self.parent.transform
        self.x_index = xindex
        self.y_index = yindex
        self.center_x = x
        self.center_y = y
        self.array = array
        self.lat = None
        self.lon = None
        self.processed = None
        self.cropped = None
        if self.parent.classifier:
            self.array = self.disassociate_one_hot()

        if self.parent.model_done:
            self.rgba = self.pimg = self.newsample = None
        else:
            self.rgba = np.transpose(self.array, (1, 2, 0)) if self.parent.banddim == 0 else self.array

            self.pimg = pimg.fromarray(self.rgba) if self.parent.resample or self.parent.needfiles else self.rgba

            self.newsample = self.pimg.resize((self.parent.resamplesizex, self.parent.resamplesizey),
                                              resample=pimg.BILINEAR) if self.parent.resample else self.pimg

        if self.parent.needfiles:
            self.png = ('row' + str(self.y_index).zfill(4)
                        + '_col' + str(self.x_index).zfill(4)
                        + '_' + self.parent.filename.replace('.tif', '') + '.png')
            if not self.parent.model_done:
                self.newsample.save(self.parent.model_input_dir + self.png)

    def disassociate_one_hot(self):
        cls_list = []
        for cls in range(self.parent.numclasses):
            zero_array = np.zeros(self.array.shape, dtype='uint8')
            zero_array[self.array == cls] = 1
            cls_list.append(zero_array)
        return np.stack(cls_list, axis=-1)

    def find_center_lat_lon(self, coord_transform):
        """
        Find the lat and lon of the center of the window.

        Returns:
            Longitude, Latitude
        """
        if self.t_form:
            centerlon = self.t_form[0] + self.t_form[1] * self.center_x
            centerlat = self.t_form[3] + self.t_form[5] * self.center_y
            self.lon, self.lat = coord_transform.TransformPoint(centerlon, centerlat, 0)[:-1]

    def collect_processed_data(self):
        """
        Collect processed images.

        """
        if self.parent.needfiles:
            pic = pimg.open(self.parent.model_output_dir + self.parent.fileprefix + self.png)
            self.processed = np.array(pic.getdata(), dtype='uint8').reshape(pic.size[0],
                                                                            pic.size[1], len(pic.getbands()))
            if not self.parent.classifier:
                self.crop_processed_data()
            else:
                self.cropped = self.processed
        else:
            self.processed = np.array([0, 1, 2, 3, 4])  # change this to reflect output from tensorflow
            self.cropped = self.processed  # change this to reflect output from tensorflow

    def crop_processed_data(self):
        """
        Crop processed images.

        """

        self.cropped = self.processed[self.parent.x_left_crop: self.parent.x_right_crop or None,
                       self.parent.y_up_crop: self.parent.y_down_crop or None, :]

class Image:
    def __init__(self, filename: str, model_done: bool = False):
        for k, v in params.items():
            setattr(self, k, v)
        self.filename = filename
        self.model_done = model_done
        if '.tif' in self.filename:
            if self.file_input_dir:
                self.img = gdal.Open(os.path.join(self.file_input_dir, filename))
            else:
                self.img = gdal.Open(filename)
            self.proj = self.img.GetProjection()
            self.transform = self.img.GetGeoTransform()
            if self.classifier:
                self.numdims = 2
                self.banddim = -1
            else:
                self.numdims = 3
                self.banddim = 0
            self.h = self.img.RasterYSize
            self.w = self.img.RasterXSize
            if self.model_done:
                self.array = None
            else:
                self.array = self.img.ReadAsArray().astype('uint8')
                self.array = self.get_bands()
                if self.classifier:
                    self.numclasses = int(self.array.max()) + 1
                else:
                    self.numclasses = None
        else:
            if self.file_input_dir:
                self.img = pd.read_csv(os.path.join(self.file_input_dir, filename))
            else:
                self.img = pd.read_csv(filename)
            self.proj = None
            self.transform = None
            self.array = self.img
            self.numdims = len(self.array.shape)
            self.banddim, self.h, self.w = self.parse_dims()
            self.array = self.get_bands()
        if self.stridex % 2:
            self.xinds = np.arange(int(np.floor(self.stridex / 2)), self.w + self.stridex - (self.w % self.stridex), self.stridex).astype(int)
        else:
            self.xinds = np.arange(self.stridex / 2, self.w + self.stridex - (self.w % self.stridex), self.stridex).astype(int)
        if self.stridey % 2:
            self.yinds = np.arange(int(np.floor(self.stridey / 2)), self.h + self.stridey - (self.h % self.stridey), self.stridey).astype(int)
        else:
            self.yinds = np.arange(self.stridey / 2, self.h + self.stridey - (self.h % self.stridey), self.stridey).astype(int)

        self.windows = self.make_windows()

    def parse_dims(self) -> list:
        """
        Parse the dimensions of the incoming array.

        This method attempts to identify which dimension of the incoming image array is the "band" dimension, if one
        exists.

        Args:
            self: Image class to parse

        Returns:
            List of the band dimension, the number of rows in the image, and the number of columns in the image.
        """

        if self.numdims < 3:
            return [-1, self.array.shape[0], self.array.shape[1]]
        else:
            banddim = np.argmin(self.array.shape)
            rowscols = [self.array.shape[dim] for dim in range(3) if dim != banddim]
            return [banddim] + rowscols

    def get_bands(self) -> np.ndarray:
        """
        Grab only the desired bands of the incoming array.

        This method subsets the full incoming array by grabbing only the bands specified in the environment variable.

        Args:
            self: Image class to parse

        Returns:
            Subset of original array
        """

        if self.banddim < 0:
            return self.array
        elif self.banddim == 0:
            return self.array[self.window_bands, ...]
        else:
            return self.array[..., self.window_bands]

    def make_windows(self) -> np.ndarray:
        """
        Output windows given image.

        This function creates a numpy ndarray of "windows" of the original image, based on the stride specified and the
        window size specified.

        Args:
            self: Image class to window

        Returns:
            Numpy ndarray of output windows.
        """
        windows = np.ndarray((self.yinds.size - self.chop_last_row,
                              self.xinds.size - self.chop_last_col), dtype=Window)
        if self.windowsizex % 2 and self.stridex % 2:
            self.radius_x_left = int(np.floor(self.windowsizex / 2))
            self.radius_x_right = int(np.floor(self.windowsizex / 2)) + 1
            self.x_left_crop = int((self.windowsizex - self.stridex) / 2)
            self.x_right_crop = int((self.windowsizex - self.stridex) / -2)
        elif self.windowsizex % 2 and not self.stridex % 2:
            self.radius_x_left = int(np.floor(self.windowsizex / 2))
            self.radius_x_right = int(np.floor(self.windowsizex / 2)) + 1
            self.x_left_crop = int(np.floor((self.windowsizex - self.stridex) / 2))
            self.x_right_crop = int(np.ceil((self.windowsizex - self.stridex) / -2))
        elif not self.windowsizex % 2 and self.stridex % 2:
            self.radius_x_left = int(self.windowsizex / 2 - 1)
            self.radius_x_right = int(self.windowsizex / 2 + 1)
            self.x_left_crop = int(np.floor((self.windowsizex - self.stridex) / 2))
            self.x_right_crop = int(np.ceil((self.windowsizex - self.stridex) / -2))
        else:
            self.radius_x_left = self.radius_x_right = int(self.windowsizex / 2)
            self.x_left_crop = int((self.windowsizex - self.stridex) / 2)
            self.x_right_crop = int((self.windowsizex - self.stridex) / -2)
        if self.windowsizey % 2 and self.stridey % 2:
            self.radius_y_up = int(np.floor(self.windowsizey / 2))
            self.radius_y_down = int(np.floor(self.windowsizey / 2)) + 1
            self.y_up_crop = int((self.windowsizey - self.stridey) / 2)
            self.y_down_crop = int((self.windowsizey - self.stridey) / -2)
        elif self.windowsizey % 2 and not self.stridey % 2:
            self.radius_y_up = int(np.floor(self.windowsizey / 2))
            self.radius_y_down = int(np.floor(self.windowsizey / 2)) + 1
            self.y_up_crop = int(np.floor((self.windowsizey - self.stridey) / 2))
            self.y_down_crop = int(np.ceil((self.windowsizey - self.stridey) / -2))
        elif not self.windowsizey % 2 and self.stridey % 2:
            self.radius_y_up = int(self.windowsizey / 2 - 1)
            self.radius_y_down = int(self.windowsizey / 2 + 1)
            self.y_up_crop = int(np.floor((self.windowsizey - self.stridey) / 2))
            self.y_down_crop = int(np.ceil((self.windowsizey - self.stridey) / -2))
        else:
            self.radius_y_up = self.radius_y_down = int(self.windowsizey / 2)
            self.y_up_crop = int((self.windowsizey - self.stridey) / 2)
            self.y_down_crop = int((self.windowsizey - self.stridey) / -2)

        if self.model_done:
            for yix, y in enumerate(self.yinds[:-1 * self.chop_last_row or None]):
                for xix, x in enumerate(self.xinds[:-1 * self.chop_last_col or None]):
                    windows[yix, xix] = Window(self, xix, x, yix, y)
        else:
            if self.banddim <= 0:
                for yix, y in enumerate(self.yinds[:-1 * self.chop_last_row or None]):
                    for xix, x in enumerate(self.xinds[:-1 * self.chop_last_col or None]):
                        padding = {}
                        x_left = x - self.radius_x_left
                        if x_left < 0:
                            padding['left'] = -1 * x_left
                            x_left = 0
                        else:
                            padding['left'] = 0
                        y_up = y - self.radius_y_up
                        if y_up < 0:
                            padding['up'] = -1 * y_up
                            y_up = 0
                        else:
                            padding['up'] = 0
                        x_right = x + self.radius_x_right
                        if x_right > self.w:
                            padding['right'] = x_right - self.w
                            x_right = self.w
                        else:
                            padding['right'] = 0
                        y_down = y + self.radius_y_down
                        if y_down > self.h:
                            padding['down'] = y_down - self.h
                            y_down = self.h
                        else:
                            padding['down'] = 0
                        if self.banddim < 0:
                            window_array = self.array[y_up: y_down, x_left: x_right]
                            if self.classifier:
                                padded_window_array = window_array
                            else:
                                padded_window_array = np.pad(window_array, [(padding['up'], padding['down']),
                                                                            (padding['left'], padding['right'])],
                                                             mode='constant')
                        else:
                            window_array = self.array[..., y_up: y_down, x_left: x_right]
                            padded_window_array = np.pad(window_array, [(0, 0),
                                                                        (padding['up'], padding['down']),
                                                                        (padding['left'], padding['right'])],
                                                         mode='constant')
                        windows[yix, xix] = Window(self, xix, x, yix, y, array=padded_window_array)
            else:
                for yix, y in enumerate(self.yinds[:-1 * self.chop_last_row or None]):
                    for xix, x in enumerate(self.xinds[:-1 * self.chop_last_col or None]):
                        padding = {}
                        x_left = x - self.radius_x_left
                        if x_left < 0:
                            padding['left'] = -1 * x_left
                            x_left = 0
                        else:
                            padding['left'] = 0
                        y_up = y - self.radius_y_up
                        if y_up < 0:
                            padding['up'] = -1 * y_up
                            y_up = 0
                        else:
                            padding['up'] = 0
                        x_right = x + self.radius_x_right
                        if x_right > self.w:
                            padding['right'] = x_right - self.w
                            x_right = self.w
                        else:
                            padding['right'] = 0
                        y_down = y + self.radius_y_down
                        if y_down > self.h:
                            padding['down'] = y_down - self.h
                            y_down = self.h
                        else:
                            padding['down'] = 0
                        window_array = self.array[y_up: y_down, x_left: x_right, ...]
                        padded_window_array = np.pad(window_array, [(padding['up'], padding['down']),
                                                                    (padding['left'], padding['right']),
                                                                    (0, 0)],
                                                     mode='constant')
                        windows[yix, xix] = Window(self, xix, x, yix, y, array=padded_window_array)
        return windows

    def remove_window_files(self, mode='input'):
        """
        Remove window pngs.

        Args:
            self: Image class to loop through.
            mode: Which pngs to remove (input to the model ("input"), processed data ("processed") or
                    both ("input_processed")
        """
        if 'input' in mode:
            for wdow in self.windows.flatten():
                os.remove(self.model_input_dir + wdow.png)
        if 'processed' in mode:
            for wdow in self.windows.flatten():
                os.remove(self.model_output_dir + self.fileprefix + wdow.png)

    def find_window_lat_lons(self):
        """
        Loop through windows and find the lat / lon of the center point.

        Args:
            self: Image class to loop through.
        """

        p1 = osr.SpatialReference()
        p1.ImportFromEPSG(4326)
        p2 = osr.SpatialReference()
        p2.ImportFromWkt(self.proj)
        coord_transform = osr.CoordinateTransformation(p2, p1)

        for wdow in self.windows.flatten():
            wdow.find_center_lat_lon(coord_transform)

    def collect_proc_windows(self):
        """
        Loop through windows and import their processed data.

        Args:
            self: Image class to loop through.
        """

        for wdow in self.windows.flatten():
            wdow.collect_processed_data()

    def reassemble_windows(self) -> np.ndarray:
        """
        Re-assemble windows to form image.

        This function assembles the output from tensorflow contained in each window back into a cohesive image.

        Args:
            self: Image class to re-assemble.

        Returns:
            Re-assembled numpy array (image).
        """

        bigcat = []
        colcount = self.windows.shape[0]
        rowcount = self.windows.shape[1]
        for row in range(rowcount):
            this_row = [wdow.cropped for wdow in self.windows[:, row]]
            bigcat.append(np.concatenate(this_row[:colcount], axis=0))
        final_img = np.concatenate(bigcat, axis=1)
        if self.orig_image_shape:
            return final_img[:self.h, :self.w, :]
        else:
            return final_img


if __name__ == "__main__":
    tifs = os.listdir(os.getcwd())
    images = [Image(tif) for tif in tifs if '.tif' in tif]
