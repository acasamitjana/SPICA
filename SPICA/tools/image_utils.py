# -*- coding: utf-8 -*-

"""
@author: Miquel Gubau Besalú and Adrià Casamitjana Díaz
"""

import math as m

PI = m.pi

import numpy as np
import PIL


class MyImage:

    def __init__(self, img, res, id_):
        self.img = img
        self.res = self.__res(res)
        self.x_size, self.y_size = self.__size()
        self.id = id_

        self.centre = None
        self.holes = None
        self.markers = None
        self.source = None

        self.processed_img = None

    @staticmethod
    def __res(res):
        res = np.asarray(res)
        if (res.size == 0):
            raise Exception("Invalid Attribute")
        elif (res.size > 2):
            raise Exception("Invalid Attribute")
        res = np.unique(res)
        if (res.size == 1):
            return res[0]
        else:
            raise Exception("Images must have the same resolution on both axis")

    def __size(self):
        return self.img[0, :].size, self.img[:, 0].size

    def image_invert(self, mode='horizontal'):
        if (mode == 'horizontal'):
            self.img = self.img[:, ::-1]
        elif (mode == 'vertical'):
            self.img = self.img[::-1, :]
        elif (mode == 'both'):
            self.img = self.img[::-1, ::-1]
        else:
            raise Exception("Invalid Mode")

    def image_resize(self, new_res):
        new_res = self.__res(new_res)
        if (self.res == new_res):
            return
        elif (self.res < new_res):
            raise Exception("New resolution must be lower than the actual one")
        step = int(round(1 / (1 - new_res / self.res)))
        self.img = np.delete(np.delete(self.img, np.s_[::step], 0), np.s_[::step], 1)
        self.x_size, self.y_size = self.__size()
        self.res = new_res

    def image_trim(self, coord, coord0=None):
        if coord0 is None:
            coord0 = [0, 0]
        self.img = self.img[coord0[1]:coord[1], coord0[0]:coord[0]]
        self.x_size, self.y_size = self.__size()


class OvoidsInfo:
    # information from the manuals, distances in mm
    options = [22, 26, 30]  # lumen center diameters
    ang_diff_holes = [22, 18, 18]  # angle difference between adjacent holes
    _dr = 6  # difference between internal and external radius
    _h = 6  # lumen center elevation

    def __init__(self, size):
        try:
            i = self.options.index(size)
        except ValueError:
            raise Exception("Invalid Ovoid Size")
        # distances in mm
        self.ovoid = self.options[i]
        self.r_int = (self.options[i] / 2)
        self.r_ext = (self.options[i] / 2 + self._dr)
        self.adj_holes = self.ang_diff_holes[i]
        self.h = self._h

        self.num_rep = 0

        self.mean_centre = None
        self.holes = {}  # with respect to the ovoid's centre, distances in pixels

    # centre of the image resulting from superimposing the radiographs
    def calc_mean_centre(self, centres):
        self.mean_centre = [0, 0]
        for k in centres:
            self.mean_centre[0] += k[0]
            self.mean_centre[1] += k[1]
        self.mean_centre[0] /= len(centres)
        self.mean_centre[1] /= len(centres)

    def add_holes(self, _hole):
        hole = np.copy(_hole)
        hole[0] -= self.mean_centre[0]
        hole[1] -= self.mean_centre[1]
        hole[1] *= -1
        ang = m.atan(hole[1] / hole[0]) * 180 / PI
        if hole[0] < 0:
            ang += 180
        elif hole[0] > 0 and hole[1] < 0:
            ang += 360
        ang = round(ang)
        if any(ang == key for key in self.holes.keys()):
            self.holes[ang].append(hole)
        else:
            self.holes.update({ang: [hole]})
        return

    def reset_holes(self):
        self.holes = {}


# add an image to an array using MyImage object
def add_image(list_, path, id_, channel=0):
    img = PIL.Image.open(path)
    list_.append(MyImage(np.copy(img.getchannel(channel)), img.info['dpi'], id_))
    return list_