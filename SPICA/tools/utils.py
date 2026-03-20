# -*- coding: utf-8 -*-

"""
@author: Miquel Gubau Besalú and Adrià Casamitjana Díaz
"""

import math as m

# import matplotlib
import matplotlib.pyplot as plt

import numpy as np

# import scipy
from scipy import ndimage
from scipy import optimize
from skimage import feature
from skimage import filters
from skimage import morphology
from sklearn.cluster import DBSCAN
from skimage import measure
from skimage.transform import resize

# CUSTOM RESOURCES
from SPICA.tools.venezia_library import INCH
from SPICA.tools.image_utils import MyImage, OvoidsInfo

# %%

""" IMAGES' LOADING FUNCTIONS """


def path_format(path):
    if path is None or len(path) == 0:
        return None
    if (path != '') and (path[-1] != '\\') and (path[-1] != '/'):
        path += '/'
    if path.count('\\') != 0:
        path = path.replace('\\', '/')
    return path


# %%

""" IMAGES' MATCHING FUNCTIONS """

def get_min_res(imgs):
    min_res = imgs[0].res
    # compare resolutions
    for k in imgs[1:]:
        if (min_res != k.res):
            if (min_res > k.res):
                min_res = k.res

    return min_res

def set_same_res(imgs, res=None):
    if res is None:
        same_res = True
        min_res = imgs[0].res
        # compare resolutions
        for k in imgs[1:]:
            if (min_res != k.res):
                same_res = False
                if (min_res > k.res):
                    min_res = k.res
        res = min_res

    # change resolution if necessary
    for k in imgs:
        if (res != k.res):
            k.image_resize(res)

    return imgs


def get_trim_param(img):
    edges_sobel = filters.sobel(img)

    grad_x = np.zeros([edges_sobel[0, :].size])
    for i in range(edges_sobel[0, :].size):
        grad_x[i] = edges_sobel[:, i].sum()
    x = np.argmax(np.abs(grad_x))

    grad_y = np.zeros([edges_sobel[:, 0].size])
    for j in range(edges_sobel[:, 0].size):
        grad_y[j] = edges_sobel[j, :].sum()
    y = np.argmax(np.abs(grad_y))

    return x, y

def get_min_fov(imgs):
    x_min, y_min = 0, 0

    # compare resolutions
    for img in imgs:
        x, y = get_trim_param(img.img)
        if x > x_min:
            x_min = x

        if y > y_min:
            y_min = y

    return x_min, y_min


def set_same_size(imgs):
    min_x = imgs[0].x_size
    min_y = imgs[0].y_size
    for k in imgs[1:]:
        if (min_x > k.x_size):
            min_x = k.x_size
        if (min_y > k.y_size):
            min_y = k.y_size
    for k in imgs:
        x = k.x_size
        y = k.y_size
        k.image_trim(coord=[x, y], coord0=[x - min_x, y - min_y])
    return imgs


# %%

""" IMAGES' PREPROCESSING FUNCTIONS """


def normalization(img):
    img = img.astype(float)
    img -= img.min()
    img /= img.max()
    return img


def gamma_contrast(img, gamma=0.5):
    """
    Parameters
    ----------
    img : array
        DESCRIPTION.
    gamma : float
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    img : array
        DESCRIPTION.
    """
    for i in range(img[0, :].size):
        for j in range(img[:, 0].size):
            img[j, i] = img[j, i] ** gamma
    return img


# %%

def gradient(img, mode='corners', th=0):
    """
    Parameters
    ----------
    img : array
        DESCRIPTION.
    mode : 'corners', 'vertical', 'horizontal'
        DESCRIPTION. The default is 'corners'.
    th : int or float
        DESCRIPTION. The default is 0.

    Returns
    -------
    grad : 2D array
        DESCRIPTION.
    """
    # calculation of the quadrants mean value
    values = []
    size_x = img[0, :].size
    size_y = img[:, 0].size
    values.append(mean_th(img[:size_y // 2, :size_x // 2], th))
    # values.append(img[:size_y//2, :size_x//2].max())
    values.append(mean_th(img[:size_y // 2, size_x - size_x // 2:size_x], th))
    # values.append(img[:size_y//2, size_x - size_x//2:size_x].max())
    values.append(mean_th(img[size_y - size_y // 2:size_y, :size_x // 2], th))
    # values.append(img[size_y - size_y//2:size_y, :size_x//2].max())
    values.append(mean_th(img[size_y - size_y // 2:size_y, size_x - size_x // 2:size_x], th))
    # values.append(img[size_y - size_y//2:size_y, size_x - size_x//2:size_x].max())

    if (mode == 'corners'):
        pass
    elif (mode == 'vertical'):
        values[0] = (values[0] + values[1]) / 2
        values[1] = values[0]
        values[2] = (values[2] + values[3]) / 2
        values[3] = values[2]
    elif (mode == 'horizontal'):
        values[0] = (values[0] + values[2]) / 2
        values[2] = values[0]
        values[1] = (values[1] + values[3]) / 2
        values[3] = values[1]
    else:
        raise Exception("Invalid Mode")

    # recalculation of the weights to have the same mean values
    average = sum(values) / len(values)
    for k in range(len(values)):
        values[k] = average / values[k]

    _grad_x = np.array([np.linspace(values[0], values[1], img[0, :].size),
                        np.linspace(values[2], values[3], img[0, :].size)])
    _grad_y = np.array([np.linspace(values[0], values[2], img[:, 0].size),
                        np.linspace(values[1], values[3], img[:, 0].size)])

    X, Y = np.linspace(0, 1, img[0, :].size), np.linspace(0, 1, img[:, 0].size)

    grad1, _ = np.meshgrid(_grad_x[0], Y)
    grad2, _ = np.meshgrid(_grad_x[1], Y)
    _, grad3 = np.meshgrid(X, _grad_y[0])
    _, grad4 = np.meshgrid(X, _grad_y[1])

    grad = grad1 + grad2 + grad3 + grad4
    grad /= 4

    return grad


def circ_mask(shape, centre, radius, mode='in'):
    """
    Parameters
    ----------
    shape : int array of size 2
        DESCRIPTION.
    centre : int array of size 2
        DESCRIPTION.
    radius : int or float
        DESCRIPTION.
    mode : 'in', 'out'
        DESCRIPTION. The default is 'in'.

    Returns
    -------
    mask : 2D array
        DESCRIPTION.
    """
    if (mode == 'in'):
        mask = np.ones(shape, dtype='int')
        fill = 0

    elif (mode == 'out'):
        mask = np.zeros(shape, dtype='int')
        fill = 1
    else:
        raise Exception("Invalid Mode")

    for i in range(mask[0, :].size):
        for j in range(mask[:, 0].size):
            if m.sqrt((i - centre[0]) ** 2 + (j - centre[1]) ** 2) < radius:
                mask[j, i] = fill
    return mask


# %%

def mean_th(img: np.ndarray, th: float, mode='bigger'):
    """
    Parameters
    ----------
    img : array
        DESCRIPTION.
    th : float
        DESCRIPTION.
    mode : 'bigger', 'lower'
        DESCRIPTION. The default is 'bigger'.

    Returns
    -------
    new_th : float
        DESCRIPTION.
    """
    # pixels are filtered
    if mode == 'bigger':
        mask = img > th

    elif mode == 'lower':
        mask = img < th

    else:
        raise Exception("Invalid Mode")

    return np.mean(img[mask])



def thresholding(img, th, inverse=False):
    """
    Parameters
    ----------
    img : int or float array
        DESCRIPTION. Image.
    th : int, float or array
        DESCRIPTION. Threshold(s) to apply.
    inverse : bool
        DESCRIPTION.

    Returns
    -------
    th_img : int array
        DESCRIPTION. Image with the threshold(s) applied.
    """
    dtype_ = 'int'
    if (inverse == False):
        v = 1
        th_img = np.zeros([img[:, 0].size, img[0, :].size], dtype=dtype_)
    else:
        v = -1
        th_img = np.ones([img[:, 0].size, img[0, :].size], dtype=dtype_)

    ths = np.asarray(th)
    for k in range(ths.size):
        if (ths.size != 1):
            th = ths[k]
        else:
            th = ths
        for i in range(img[0, :].size):
            for j in range(img[:, 0].size):
                if (img[j, i] >= th):
                    th_img[j, i] += v
    return th_img


def ensemble_centroid(timc, value):
    """
    Parameters
    ----------
    timc : int array
        DESCRIPTION.
    value : int
        DESCRIPTION.

    Returns
    -------
    centroid : int array of size 2
        DESCRIPTION.
    """
    # MATRIX WITH THE POINTS THAT HAVE THE VALUE SPECIFIED
    # the matrix is created
    d = np.empty([np.count_nonzero(timc == value), 2], dtype='int')
    # the coordinates of each point are saved
    k = 0
    for i in range(timc[0, :].size):
        for j in range(timc[:, 0].size):
            if timc[j, i] == value:
                d[k, 0] = i
                d[k, 1] = j
                k += 1
    # CENTROID VECTOR
    centroid = [0, 0]  # np.zeros([2], dtype='int')
    # the number of points is taken
    n = d[:, 0].size
    # the coordinates of each point are added
    for i in range(n):
        centroid[0] += d[i, 0]
        centroid[1] += d[i, 1]
    # the mean is calulated
    centroid[0] = round(centroid[0] / n)
    centroid[1] = round(centroid[1] / n)
    return centroid


# %%

# arranges the data to cluster
def _pre_clustering(img, value):
    data = np.empty([np.count_nonzero(img == value), 2], dtype='int')
    k = 0
    for i in range(img[0, :].size):
        for j in range(img[:, 0].size):
            if img[j, i] == value:
                data[k, 0] = i
                data[k, 1] = j
                k += 1
    return data


# rearranges the information from the clustered data
def _post_clustering(img, data, labels, centroids=True):
    for k in range(data[:, 0].size):
        img[data[k, 1], data[k, 0]] += labels[k]
    if (centroids == True):
        centroids = []
        for k in range(1, img.max() + 1):
            centroids.append(ensemble_centroid(img, k))
        return img, centroids
    else:
        return img


# calculates the circular representation with a centre and a radius
def _circ_func(centre, radius):
    angle = np.linspace(0, 2 * m.pi, 360)
    x = centre[0] + radius * np.sin(angle)
    y = centre[1] + radius * np.cos(angle)
    return np.array([x, y])


# calculates the centre distances to minimize
def _radius_func(centre, coords):
    radius = np.zeros([coords[:, 0].size])
    for k in range(coords[:, 0].size):
        radius[k] = m.sqrt((coords[k, 0] - centre[0]) ** 2 + (coords[k, 1] - centre[1]) ** 2)
    return radius


def _diff1(centre, coords):
    points = coords[:, 0].size
    mean_radius = _radius_func(centre, coords).sum()
    mean_radius /= points
    radius = np.zeros([points])
    for k in range(points):
        radius[k] = m.sqrt((coords[k, 0] - centre[0]) ** 2 + (coords[k, 1] - centre[1]) ** 2)
        radius[k] -= mean_radius
        radius[k] = abs(radius[k])
    return radius.sum() / points


def _diff2(centre, coords):
    points = coords[:, 0].size
    mean_radius = _radius_func(centre, coords).sum()
    mean_radius /= points
    radius = np.zeros([points])
    for k in range(points):
        radius[k] = m.sqrt((coords[k, 0] - centre[0]) ** 2 + (coords[k, 1] - centre[1]) ** 2)
        radius[k] -= mean_radius
        radius[k] = abs(radius[k])
    return radius


def _optim_centre(fit_coords, centre, plot=False):
    fit_coords = np.array(fit_coords)
    points = fit_coords[:, 0].size

    if (plot == True):
        plt.figure()

        plt.scatter(centre[0], centre[1], color='tab:orange', marker='+')

    r0 = _radius_func(centre, fit_coords).sum() / points
    c0 = _circ_func(centre, r0)

    if (plot == True):
        plt.plot(c0[0], c0[1], color='tab:orange')
        plt.scatter(centre[0], centre[1], color='tab:green', marker='+')

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    centre1 = optimize.minimize(_diff1, centre, args=(fit_coords))

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
    # centre2 = optimize.leastsq(_diff2, centre, args=(fit_coords))

    new_centre = centre1.x
    # new_centre = centre2[0]

    r = _radius_func(new_centre, fit_coords).sum() / points
    c = _circ_func(new_centre, r)

    if (plot == True):
        plt.plot(c[0], c[1], color='tab:green')
        plt.scatter(fit_coords[:, 0], fit_coords[:, 1], marker='x')

        plt.tight_layout()

    return new_centre, r

def image_histogram_equalization(image: np.ndarray, number_bins: int=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    M, m = np.max(image), np.min(image)

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image_equalized = (image_equalized / 255)*(M - m) + m

    return image_equalized.reshape(image.shape)#, cdf


def crop_label(mask: np.ndarray, margin: int=10, threshold: int=0) -> tuple:
    ndim = len(mask.shape)
    if isinstance(margin, int):
        margin=[margin]*ndim

    crop_coord = []
    idx = np.where(mask>threshold)
    for it_index, index in enumerate(idx):
        clow = max(0, np.min(idx[it_index]) - margin[it_index])
        chigh = min(mask.shape[it_index], np.max(idx[it_index]) + margin[it_index])
        crop_coord.append([clow, chigh])

    mask_cropped = mask[
                   crop_coord[0][0]: crop_coord[0][1],
                   crop_coord[1][0]: crop_coord[1][1],
                   ]

    return mask_cropped, crop_coord

def compute_dice_coef(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)

    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# %%
""" RADIOGRAPH PROCESSING """


def processRAD(image: MyImage, ovoids: OvoidsInfo, figures: bool = False, eq_hist: bool = False) -> MyImage:
    '''
    Radiograph image processing function. It follows the following pipeline:
    - Gaussian blurring of 1mm
    - Applicator extraction and rough center computation
    - Holes extraction by geometric contraints (center and ovoid radius) and DBSCAN
    - Update the center calculation with the computed holes.

    Parameters
    ----------
    image : MyImage
        Radiograph image to be processed
    ovoids : OvoidsInfo
        Information about the ovoids used.
    figures : boolean, optional
        Show intermediate results of the processing steps (default: False)
    eq_hist : boolean, optional
        Equalize the histogram prior to the other processing steps (default: False)
    Returns
    -------
    image : MyImage
         Radiograph image processed (with center, radius and preprocessed image stored).
    '''



    # according to the size of the details, 1 mm
    _detail_size = round(image.res / INCH)
    # low-pass filter
    f_un = np.ones([_detail_size, _detail_size]) / _detail_size ** 2

    rad_orig = np.copy(image.img)
    rad_hist = np.copy(image.img)
    rad_hist = image_histogram_equalization(rad_hist)

    rad_holes_global = np.zeros_like(rad_orig)
    for rad in [rad_orig, rad_hist]:
        rad = ndimage.convolve(rad, f_un)

        # pixel values correction
        # rad = gamma_contrast(rad)

        _histo = ndimage.histogram(rad, 0, 1, 256)

        # thresholds
        ths_rad = []
        ths_rad += [mean_th(rad, np.argmax(_histo) / 256, mode='lower')]
        ths_rad += [mean_th(rad, ths_rad[0], mode='lower')]

        # CENTRE CALCULATION
        # first approximation
        rad_holes = thresholding(rad, ths_rad[1], inverse=True)
        centre0 = ensemble_centroid(rad_holes, 0)

        # background elimination
        rad_holes_A = rad_holes * circ_mask(rad_holes.shape, centre0, ovoids.r_ext * image.res / INCH, mode='out')
        # centre elimination
        rad_holes_B = rad_holes_A * circ_mask(rad_holes_A.shape, centre0, ovoids.r_int * image.res / INCH)



        # clustering parameters
        # radius of every point bubble
        _eps = 0.9 * 1 * image.res / INCH  # 0.9*hole radius, 1 mm
        # minimum samples in the bubble
        _min_samples = int(0.5 * m.pi * _eps ** 2)  # 0.5*hole area

        # PROVA:
        hole_mask = 1-circ_mask([2*15, 2*15], [15, 15], 15)
        all_blobs, num_blobs = measure.label(rad_holes_B, connectivity=2, return_num=True)
        remove_idx = []
        for it_blob in range(1, num_blobs+1):
            mask_blob, _ = crop_label(all_blobs == it_blob, margin=1)
            mask_blob = resize(mask_blob, hole_mask.shape)
            dice = compute_dice_coef(mask_blob, hole_mask)
            if dice < 0.75:
                remove_idx += [it_blob]

        for ridx in remove_idx:
            all_blobs[all_blobs == ridx] = 0

        rad_holes_global += (all_blobs > 0)
        # FI PROVA

    rad_holes_global = rad_holes_global > 0
    # clustering
    # array with the coordinates of the pixels which correspond to the needle holes
    _holes_px = _pre_clustering(rad_holes_global, 1)

    # # the unclustered points are considered noise, with values of -1
    _dbscan_res = DBSCAN(eps=_eps, min_samples=_min_samples).fit(_holes_px)
    _dbscan_labels = _dbscan_res.labels_

    rad_holes_post, coord_holes = _post_clustering(rad_holes_global.astype('int'), _holes_px, _dbscan_labels)

    centre, _r = _optim_centre(np.array([[k[0], k[1]] for k in coord_holes]), [centre0[0], centre0[1]])
    # _c = _circ_func([centre[0], centre[1]], _r)

    image.centre = centre
    image.holes = coord_holes

    if (figures == True):

        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(rad, cmap='gray')
        ax[0, 1].imshow(thresholding(rad, ths_rad[0]), cmap='gray')
        ax[0, 2].imshow(thresholding(rad, ths_rad[1]), cmap='gray')
        ax[1, 0].imshow(rad_holes > 0, cmap='gray')
        ax[1, 1].imshow(rad_holes_A > 0, cmap='gray')
        ax[1, 2].imshow(rad_holes_global > 0, cmap='gray')

        ax[0, 0].scatter(centre0[0], centre0[1], color='black', marker='+')
        ax[0, 0].scatter(centre[0], centre[1], color='tab:red', marker='+')
        ax[0, 0].scatter([k[0] for k in coord_holes], [k[1] for k in coord_holes], color='tab:red')

        plt.tight_layout()

        # plt.figure()
        # plt.plot(np.linspace(0, 1, 256), _histo)
        # plt.vlines(ths_rad, 0, _histo.max(), color='red')
        # plt.tight_layout()

    # image processing to display the results
    image.processed_img = feature.canny(image.img)
    image.processed_img = morphology.binary_dilation(image.processed_img)
    #image.processed_img = morphology.diameter_opening(image.processed_img, _detail_size)
    image.processed_img = morphology.area_opening(image.processed_img, _detail_size**2)
    #image.processed_img = morphology.remove_small_objects(image.processed_img, _detail_size**2)
    #image.processed_img = morphology.erosion(image.processed_img)
    image.processed_img = morphology.thin(image.processed_img)
    image.processed_img = normalization(image.processed_img)

    return image

def processRAD_updated(image: MyImage, ovoids: OvoidsInfo, figures: bool = False, eq_hist: bool = False) -> MyImage:
    '''
    Radiograph image processing function. It follows the following pipeline:
    - Gaussian blurring of 1mm
    - Applicator extraction and rough center computation
    - Holes extraction by geometric contraints (center and ovoid radius) and DBSCAN
    - Update the center calculation with the computed holes.

    Parameters
    ----------
    image : MyImage
        Radiograph image to be processed
    ovoids : OvoidsInfo
        Information about the ovoids used.
    figures : boolean, optional
        Show intermediate results of the processing steps (default: False)
    eq_hist : boolean, optional
        Equalize the histogram prior to the other processing steps (default: False)
    Returns
    -------
    image : MyImage
         Radiograph image processed (with center, radius and preprocessed image stored).
    '''

    rad = np.copy(image.img)

    _detail_size = round(image.res / INCH)  # according to the size of the details, 1 mm
    # low-pass filter
    f_un = np.ones([_detail_size, _detail_size]) / _detail_size ** 2

    if eq_hist:
        rad = image_histogram_equalization(rad)

    rad = ndimage.convolve(rad, f_un)

    # pixel values correction
    # rad = gamma_contrast(rad)

    _histo = ndimage.histogram(rad, 0, 1, 256)

    # thresholds
    ths_rad = []
    ths_rad += [mean_th(rad, np.argmax(_histo) / 256, mode='lower')]
    ths_rad += [mean_th(rad, ths_rad[0], mode='lower')]

    # CENTRE CALCULATION

    # first approximation
    rad_holes = thresholding(rad, ths_rad[1], inverse=True)
    centre0 = ensemble_centroid(rad_holes, 0)

    # background elimination
    rad_holes_A = rad_holes * circ_mask(rad_holes.shape, centre0, ovoids.r_ext * image.res / INCH, mode='out')
    # centre elimination
    rad_holes_B = rad_holes_A * circ_mask(rad_holes_A.shape, centre0, ovoids.r_int * image.res / INCH)

    # array with the coordinates of the pixels which correspond to the needle holes
    _holes_px = _pre_clustering(rad_holes_B, 1)

    # clustering parameters
    # radius of every point bubble
    _eps = 0.9 * 1 * image.res / INCH  # 0.9*hole radius, 1 mm
    # minimum samples in the bubble
    _min_samples = int(0.5 * m.pi * _eps ** 2)  # 0.5*hole area

    # clustering
    # the unclustered points are considered noise, with values of -1
    _dbscan_res = DBSCAN(eps=_eps, min_samples=_min_samples).fit(_holes_px)
    _dbscan_labels = _dbscan_res.labels_

    rad_holes_post, coord_holes = _post_clustering(rad_holes_B, _holes_px, _dbscan_labels)

    centre, _r = _optim_centre(np.array([[k[0], k[1]] for k in coord_holes]), [centre0[0], centre0[1]])
    # _c = _circ_func([centre[0], centre[1]], _r)

    image.centre = centre
    image.holes = coord_holes

    if (figures == True):

        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(image.img, cmap='gray')
        ax[0, 1].imshow(thresholding(rad, ths_rad[0]), cmap='gray')
        ax[0, 2].imshow(thresholding(rad, ths_rad[1]), cmap='gray')
        ax[1, 0].imshow(rad_holes > 0, cmap='gray')
        ax[1, 1].imshow(rad_holes_A > 0, cmap='gray')
        ax[1, 2].imshow(rad_holes_B > 0, cmap='gray')

        ax[0, 0].scatter(centre0[0], centre0[1], color='black', marker='+')
        ax[0, 0].scatter(centre[0], centre[1], color='tab:red', marker='+')
        ax[0, 0].scatter([k[0] for k in coord_holes], [k[1] for k in coord_holes], color='tab:red')

        plt.tight_layout()

        # plt.figure()
        # plt.plot(np.linspace(0, 1, 256), _histo)
        # plt.vlines(ths_rad, 0, _histo.max(), color='red')
        # plt.tight_layout()

    # image processing to display the results
    image.processed_img = feature.canny(image.img)
    image.processed_img = morphology.binary_dilation(image.processed_img)
    #image.processed_img = morphology.diameter_opening(image.processed_img, _detail_size)
    image.processed_img = morphology.area_opening(image.processed_img, _detail_size**2)
    #image.processed_img = morphology.remove_small_objects(image.processed_img, _detail_size**2)
    #image.processed_img = morphology.erosion(image.processed_img)
    image.processed_img = morphology.thin(image.processed_img)
    image.processed_img = normalization(image.processed_img)

    return image

# %%

""" AUTORADIOGRAPH PROCESSING """


# calculates the threshold of the autoradiographs
def _autorad_threshold(imc: np.ndarray) -> float:
    # array with the size of the image perimeter with its values
    contorn = np.empty([2 * imc[:, 0].size + 2 * imc[0, :].size - 4])
    for i in range(imc[:, 0].size):
        contorn[i] = imc[i, 0]
        contorn[imc[:, 0].size + i] = imc[i, -1]
        for j in range(imc[0, :].size - 2):
            contorn[2 * imc[:, 0].size + 2 * j] = imc[0, j + 1]
            contorn[2 * imc[:, 0].size + 2 * j + 1] = imc[-1, j + 1]
    # the lowest pixel value is searched and used as threshold
    threshold = contorn.min()
    return threshold


def _weighted_centroid(image: np.ndarray, mask: np.ndarray) -> list:
    w_i = (1-image) * mask

    w = w_i.sum() # masked image, where the background has probability = 0 and the foreground > 0
    n = np.count_nonzero(image)
    if w == 0:
        return None, None

    x = 0
    y = 0
    for i in range(len(image[0, :])):
        for j in range(len(image[:, 0])):
            x += i * w_i[j, i]
            y += j * w_i[j, i]
    x /= w
    y /= w

    # deviation: all foreground pixels weighted by the intensity of the source
    dy, dx = 0, 0
    for i in range(len(image[0, :])):
        for j in range(len(image[:, 0])):
            dx += (i - x) ** 2 * w_i[j, i] / w
            dy += (j - y) ** 2 * w_i[j, i] / w

    # uncertainty measured as the deviation from the centroid in each direction
    dx = m.sqrt(dx / n)
    dy = m.sqrt(dy / n)
    # dx = m.sqrt((dx - x ** 2) / n)
    # dy = m.sqrt((dy - y ** 2) / n)

    return [float(x), float(y), float(dx), float(dy)]


def processAUTORAD(img: np.ndarray, eq_hist: bool=False) -> MyImage:
    image = img.img
    if eq_hist:
        image = image_histogram_equalization(image)

    image = normalization(image)
    th = _autorad_threshold(image)
    th_image = thresholding(image, th, inverse=True)
    # _th_image = th_image*image
    # centroid = ensemble_centroid(th_image, 1)
    centroid = _weighted_centroid(image, th_image)
    img.source = centroid
    img.processed_img = th_image
    return img




