# -*- coding: utf-8 -*-

"""
@author: Miquel Gubau Besalú 
"""
# python imports
import os
from tkinter import filedialog

# external libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# project libraries
from SPICA.tools.venezia_library import INCH
from SPICA.tools.image_utils import OvoidsInfo, add_image
import SPICA.tools.utils as utils

#%%

""" PARAMETERS LOADING """

# APPLICATOR SIZE
while True:
    try:
        _ovoid = int(input("SIZE [22/26/30]: "))
        if _ovoid == 22:
            from SPICA.tools.venezia_library import coord_22libraries as coord_libraries
            from SPICA.tools.venezia_library import coord_22holes as coord_holes

            break
        elif _ovoid == 26:
            from SPICA.tools.venezia_library import coord_26libraries as coord_libraries
            from SPICA.tools.venezia_library import coord_26holes as coord_holes

            break
        elif _ovoid == 30:
            from SPICA.tools.venezia_library import coord_30libraries as coord_libraries
            from SPICA.tools.venezia_library import coord_30holes as coord_holes

            break
        else:
            raise ValueError()
    except ValueError:
        print("incorrect")

ovoids = OvoidsInfo(_ovoid)

# ANALYSIS MODE
while True:
    try:
        _method = input("MODE [QA/COMMISSIONING/CUSTOM]: ")
        if _method == "QA":
            from SPICA.tools.venezia_library import path_markers as path_param

            break
        elif _method == "COMMISSIONING":
            from SPICA.tools.venezia_library import path_COMMISSIONING as path_param

            break
        elif _method == "CUSTOM":
            _distal = float(input("most distal position (mm): "))
            _step = float(input("step (mm): "))
            path_param = {'distal': _distal, 'step': _step, 'step0': 0}
            # step0 is the markers' shift of the second channel due to the double marker
            break
        else:
            raise ValueError()
    except ValueError:
        print("incorrect")

# BACKGROUND IMAGES
# searches the provided default radiographs
_rad_existence = 0
_dflt_path = os.path.join(os.getcwd(), 'data', 'default_rads')
_dflt_files = os.listdir(_dflt_path)
_dflt_rads = []
for _f in _dflt_files:
    if (_f[0:2] == str(ovoids.ovoid)):
        _dflt_rads.append(os.path.join(_dflt_path, _f))
        _rad_existence += 1

# %%

""" IMAGES LOADING """

# asking for images' directory
_dir = utils.path_format(filedialog.askdirectory(title="Select a directory with radiograph images"))
_rad_dir_files = os.listdir(_dir)

_dir = utils.path_format(filedialog.askdirectory(title="Select a directory with autoradiograph images"))
_auto_dir_files = os.listdir(_dir)

results_dir = utils.path_format(filedialog.askdirectory(title="Select a directory to store the results"))

print("Searching images...")

# list of the images' path that are going to be loaded
_paths = []
# list with the channel and position of the autoradiographs
# for the case of the radiographs these have the 'rad' tag
_imgs_id = []

# RADIOGRAPHS search
for _f in _rad_dir_files:
    if (_f[0:3] == 'rad'):
        _paths.append(_dir + _f)
        _imgs_id.append('rad')

# existance check
if (len(_paths) == 0):
    if (_rad_existence != 0):
        print("Using default radiographs")
        for _k in _dflt_rads:
            _paths.append(_k)
            _imgs_id.append('rad')
    else:
        raise Exception("No radiographs found")

_rad_rep = len(_paths)
ovoids.num_rep = _rad_rep

# AUTORADIOGRAPHS search
for _f in _auto_dir_files:
    if (_f[0] == 'C') and (_f[2] == 'P'):
        _paths.append(_dir + _f)
        _imgs_id.append([int(_f[1]), int(_f[3:5])])  # [channel, position]

# existance check
if (len(_paths) == ovoids.num_rep):
    raise Exception("No autoradiographs found")

# %%


print("Loading images...")
# list of images
imgs = []

for _k in range(len(_paths)):
    # the red channel is taken when the image is imported
    imgs = add_image(imgs, _paths[_k], _imgs_id[_k])

# invert horizontally
for _k in imgs:
    _k.image_invert()

# set the same resolution
imgs = utils.set_same_res(imgs)

# convert coordinates expressed in milimeters to pixels
for _key in coord_holes.keys():
    coord_holes[_key][0] *= imgs[0].res / INCH
    coord_holes[_key][1] *= imgs[0].res / INCH
    coord_holes[_key][2] *= imgs[0].res / INCH
for _key in coord_libraries.keys():
    coord_libraries[_key][0] *= imgs[0].res / INCH
    coord_libraries[_key][1] *= imgs[0].res / INCH
    coord_libraries[_key][2] *= imgs[0].res / INCH
""""""
# trim the working area
_margin = int(imgs[0].res / INCH)  # 1 mm margin to avoid edges artifacts
for _k in imgs:
    # search the film corner with Sobel edges detection
    _x, _y = utils.get_trim_param(_k.img)
    _k.image_trim([_x - _margin, _y - _margin])

# adjust the size to be the same, with the corner found as reference
imgs = utils.set_same_size(imgs)

# %%

""" RADIOGRAPHS PROCESSING """

print("Processing radiographs...")

for _k in range(ovoids.num_rep):
    # pixel values normalization
    imgs[_k].img = utils.normalization(imgs[_k].img)
    # gradient filter application
    # imgs[_k].img = utils.normalization(imgs[_k].img*utils.gradient(imgs[_k].img))
    # radiograph processing
    imgs[_k] = utils.processRAD(imgs[_k], ovoids)

# %%

# ovoids centre calculation using the mean value of the radiographs' measures
_centres = []
for _k in range(ovoids.num_rep):
    _centres.append(imgs[_k].centre)
ovoids.calc_mean_centre(_centres)
"""
# holes divergence correction to match the theoretical
for _k in range(ovoids.num_rep):
    for _l in range(len(imgs[_k].holes)):
        imgs[_k].holes[_l][0] += (imgs[_k].centre[0] - imgs[_k].holes[_l][0])*ovoids.h/H
        imgs[_k].holes[_l][1] += (imgs[_k].centre[1] - imgs[_k].holes[_l][1])*ovoids.h/H
"""
# centre correction by adjusting the theoretical holes to those detected
ovoids.reset_holes()  # reset for development purposes
for _k in range(ovoids.num_rep):
    for _ele in imgs[_k].holes:
        ovoids.add_holes(_ele)

_diff = [0, 0]
_n = 0
for _k in coord_holes.keys(): # iterate over theoretical hole coordinates
    for _l in ovoids.holes.keys(): # iterate over detected holes
        diff = np.abs(_l - _k)
        if diff > 180:
            diff = 360 - diff

        if diff < ovoids.adj_holes / 2: # if the detected angle and the theoretical hole coincide, update the difference in coordinates.
            for _ele in ovoids.holes[_l]:
                _diff[0] += _ele[0] - coord_holes[_k][0]
                _diff[1] += _ele[1] - coord_holes[_k][1]
                _n += 1

_diff[0] /= _n
_diff[1] /= _n

ovoids.mean_centre[0] += _diff[0]
ovoids.mean_centre[1] -= _diff[1]

# %%

""" AUTORADIOGRAPHS PROCESSING """

print("Processing autoradiographs...")

for _k in range(ovoids.num_rep, len(imgs)):
    imgs[_k] = utils.processAUTORAD(imgs[_k])

# %%

""" RESULTS """

# background image to display the results
background = np.zeros([imgs[0].y_size, imgs[0].x_size])
for _k in range(ovoids.num_rep):
    background += imgs[_k].processed_img
background /= ovoids.num_rep

# figure to show the results
plt.figure()

plt.imshow(background, cmap='gray')
# plt.imshow(imgs[0].img, cmap='gray')

_pos_dict = {'channel': [], 'distance': [], 'c_x': [], 'c_y': [], 'uA_x': [], 'uA_y': []}

for _k in range(ovoids.num_rep, len(imgs)):
    # data preparation
    _pos = [imgs[_k].id[0], path_param['distal']]
    if (imgs[_k].id[0] == 1):
        _pos[1] -= (imgs[_k].id[1] - 1) * path_param['step']
    elif (imgs[_k].id[0] == 2) & (imgs[_k].id[1] == 2):
        _pos[1] -= path_param['step0']
    elif (imgs[_k].id[0] == 2) & (imgs[_k].id[1] > 2):
        _pos[1] -= (imgs[_k].id[1] - 2) * path_param['step'] + path_param['step0']

    for _l in imgs[_k].source:
        _pos.append(_l)

    # changes the coordinates system
    _pos[2] -= ovoids.mean_centre[0]
    _pos[3] -= ovoids.mean_centre[1]
    _pos[3] = -_pos[3]

    # passes the coordinates in pixels to milimeters
    for _l in range(2, len(_pos)):
        _pos[_l] *= INCH / imgs[0].res

    # displays the positions of the source
    plt.scatter(imgs[_k].source[0], imgs[_k].source[1], marker='+', color='red')
    # displays only the libraries positions that have been studied
    plt.scatter(ovoids.mean_centre[0] + coord_libraries[tuple(_pos[0:2])][0],
                ovoids.mean_centre[1] - coord_libraries[tuple(_pos[0:2])][1], marker='+', color='green')

    # writes the data
    _pos_dict['channel'] += [_pos[0]]
    _pos_dict['distance'] += [_pos[1]]
    _pos_dict['c_x'] += [_pos[2]]
    _pos_dict['c_y'] += [_pos[3]]
    _pos_dict['uA_x'] += [_pos[4]]
    _pos_dict['uA_y'] += [_pos[5]]

# save data
_pos_df = pd.DataFrame(_pos_dict)
_pos_df.to_csv(os.path.join(results_dir, "coords" + str(ovoids.ovoid) + ".csv"))

# displays the reference elements
for _k in range(ovoids.num_rep):
    # measured centres
    # plt.scatter(imgs[_k].centre[0], imgs[_k].centre[1], color='red')
    # measured holes
    plt.scatter([imgs[_k].holes[_l][0] for _l in range(len(imgs[_k].holes))],
                [imgs[_k].holes[_l][1] for _l in range(len(imgs[_k].holes))], color='red')
# mean centre
plt.scatter(ovoids.mean_centre[0], ovoids.mean_centre[1], color='green')
# theoretical holes
for _key in coord_holes.keys():
    plt.scatter(ovoids.mean_centre[0] + coord_holes[_key][0], ovoids.mean_centre[1] - coord_holes[_key][1],
                color='green')

plt.savefig(os.path.join(results_dir, str(_method) + '.' + str(_ovoid) + '.png'))
plt.close()