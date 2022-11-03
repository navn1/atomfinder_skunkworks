import numpy as np
from scipy import ndimage
from PIL import Image
import math
from pylab import *
from pynverse import inversefunc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from sklearn.metrics import r2_score


# define functions to calculate recall, precision, and mean position deviation by comparing model predicted positions
# to ground truth positions.
def getScores(predictpos_x, predictpos_y, truthpos_x, truthpos_y, truthnum, pxsize):
    truepoints = []
    pos_se = 0
    tp = 0
    tolerance = np.round((0.5 / pxsize) ** 2, 1)  # the final unit is pixel^2

    for i in range(predictpos_x.size):
        for j in range(truthpos_x.size):
            square_error = (predictpos_x[i] - truthpos_x[j]) ** 2 + (predictpos_y[i] - truthpos_y[j]) ** 2
            if square_error < tolerance:
                truepoints.append((predictpos_y[i], predictpos_x[i]))
                pos_se = pos_se + square_error
                tp = tp + 1
    try:
        rmse = math.sqrt(pos_se / tp)
    except:
        rmse = 10
        recall = 0
        precision = 0
        print('No TP. ')

    recall = tp / truthnum
    if predictpos_x.shape[0] != 0:
        precision = tp / predictpos_x.size
    else:
        precision = 0

    print('position deviation = ', rmse * pxsize, 'angstrom, recall = ', tp, '/', truthnum, ' = ', recall,
          ', precision = ', tp, '/', predictpos_x.size, ' = ', precision)
    scores = (rmse * pxsize, recall, precision)
    return scores


def find_com(image_data: np.ndarray) -> np.ndarray:
    """
    Find atoms via center of mass methods
    Args:
        image_data (2D numpy array):
            2D image (usually an output of neural network)
    """
    labels, nlabels = ndimage.label(image_data)
    coordinates = np.array(
        ndimage.center_of_mass(
            image_data, labels, np.arange(nlabels) + 1))
    coordinates = coordinates.reshape(coordinates.shape[0], 2)
    return coordinates


def map8bit(data):
    return ((data - data.min()) / (data.max() - data.min()) * 255).astype('int8')


def img_resize(imgdata, change_size):
    ori_image = Image.fromarray(map8bit(imgdata), 'L')
    width, height = ori_image.size

    if change_size == 1:
        ori_content = ori_image
    elif (change_size > 1):
        # Upsample using bicubic
        ori_content = ori_image.resize((int(width * change_size), int(height * change_size)), Image.BICUBIC)
    else:
        # downsample using bilinear
        ori_content = ori_image.resize((int(width * change_size), int(height * change_size)), Image.BILINEAR)
    return np.array(ori_content)


# define function to delete predicted atomic column positions near the edges of the image
def delete_edge_predictions(edges, shape, predictpos_x, predictpos_y):
    '''
  edges:[left, right, top, bottom] as percentage of the whole image size
  '''
    dellistx = []
    for i, coordx in enumerate(predictpos_x):
        dis = abs(coordx - shape[1]) / shape[1]
        if dis > (1 - edges[0]) or dis < edges[1]:
            dellistx.append(i)
    dellisty = []
    for i, coordy in enumerate(predictpos_y):
        dis = abs(coordy - shape[0]) / shape[0]
        if dis > (1 - edges[2]) or dis < edges[3]:
            dellisty.append(i)

    dellist = dellistx + dellisty
    dellist = np.unique(np.asarray(dellist))
    try:
        predictpos_x = np.delete(predictpos_x, dellist)
        predictpos_y = np.delete(predictpos_y, dellist)
    except:
        print('You need to re-adjust your edge values for the most accurate measurement\n')

    return (predictpos_x, predictpos_y)


def poly3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def ssqrt(x, a, b, c):
    return np.sqrt(a + b * x) + c


def expt(x, a, b, c):
    return a * exp(-x / b) + c


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)


def logf(x, a, b, c):
    return a * log(-x / b) + c


def visualscore_pchip(paramlist, scorelist, xytitlenames, scorelimit, ylim):
    '''
     function to plot the scores vs parameter using PCHIP interpolation
     the paramlist should be strictly increasing sequence
  '''
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.canvas.draw()
    ax.scatter(paramlist, scorelist, c='b', label='Simu')

    min = np.round(np.min(paramlist), 2)
    max = np.round(np.max(paramlist), 2)

    x = np.linspace(min, max, num=500)
    y = pchip_interpolate(paramlist, scorelist, x)
    ax.plot(x, y, label="pchip interpolation")

    for i in range(y.shape[0]):
        if abs((y[i] - scorelimit)) < 0.01 * scorelimit:
            cutoff = x[i]
            cutoff = round(cutoff, 2)

    dot = ax.scatter(cutoff, scorelimit, c='r')
    dot.set_label('\nCutoff = ' + str(cutoff))
    ax.legend()

    plt.xlabel(xytitlenames[0], fontsize=16)
    plt.ylabel(xytitlenames[1], fontsize=16)
    plt.title(xytitlenames[2], fontsize=18)
    # labels = [' ','0.5', '1', '2', '4', '6', '8', '10', '12']
    # ax.set_xticklabels(labels)
    plt.grid('on')
    ax.tick_params(direction='in')

    ax.set_ylim(ylim)
    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    plt.show()


def visualscore_nonlinear(domain, obj, paramlist, scorelist, xytitlenames, scorelimit, ylim):
    '''
  Available functions: ssqrt, expt, sigmoid
  May need change: 
  - domain for calculating cutoff, the narrower the better
  - initial guess p0 or not

  '''
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.canvas.draw()
    ax.scatter(paramlist, scorelist, label='Cases')

    p0 = [np.max(scorelist), np.median(paramlist), 1, np.min(scorelist)]  # this is an mandatory initial guess

    popt, pcov = curve_fit(obj, paramlist, scorelist, method='trf', p0=p0)
    print('coefs:', popt)
    r2 = np.round(r2_score(np.asarray(scorelist), obj(np.asarray(paramlist), *popt)), 3)

    minum = np.round(np.min(paramlist), 3)
    maximum = np.round(np.max(paramlist), 3)

    func = lambda x: obj(x, *popt)
    cutoff = inversefunc(func, y_values=[scorelimit], domain=domain)[0]  ###############
    cutoff = round(cutoff, 3)
    ax.plot(np.linspace(minum, maximum, 100), obj(np.linspace(minum, maximum, 100), *popt), '--', c='r',
            label='R\N{SUPERSCRIPT TWO}: ' + str(r2))

    dot = ax.scatter(cutoff, scorelimit, c='r')
    dot.set_label('\nCutoff = ' + str(cutoff))
    ax.legend()

    plt.xlabel(xytitlenames[0], fontsize=16)
    plt.ylabel(xytitlenames[1], fontsize=16)
    plt.title(xytitlenames[2], fontsize=18)
    # labels = [' ','0.5', '1', '2', '4', '6', '8', '10', '12']
    # ax.set_xticklabels(labels)
    plt.grid('on')
    ax.tick_params(direction='in')

    ax.set_ylim(ylim)
    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.show()


def visualscore_nonlinear_compare(domain, obj, paramlist, scorelist, xytitlenames, scorelimit, eparamlist, escorelist,
                                  ylim):
    '''
  Available functions: ssqrt, expt, sigmoid
  May need change: 
  - domain for calculating cutoff, the narrower the better
  - initial guess p0 or not

  '''
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.canvas.draw()
    ax.scatter(paramlist, scorelist, c='b', label='Ziatdinov', s=1)
    ax.scatter(eparamlist, escorelist, c='g', label='new')
    try:
        p0 = [max(scorelist), np.median(paramlist), 1, min(scorelist)]  # this is an mandatory initial guess

        popt, pcov = curve_fit(obj, paramlist, scorelist, method='trf', p0=p0)
        print('coefs:', popt)
        r2 = np.round(r2_score(np.asarray(scorelist), obj(np.asarray(paramlist), *popt)), 3)

        minum = np.round(np.min(paramlist), 3)
        maximum = np.round(np.max(paramlist), 3)

        func = lambda x: obj(x, *popt)
        cutoff = inversefunc(func, y_values=[scorelimit], domain=domain)[0]  ###############
        cutoff = round(cutoff, 3)
        ax.plot(np.linspace(minum, maximum, 100), obj(np.linspace(minum, maximum, 100), *popt), '--', c='r',
                label='R\N{SUPERSCRIPT TWO}: ' + str(r2))

        dot = ax.scatter(cutoff, scorelimit, c='r', marker='^')
        dot.set_label('\nCutoff = ' + str(cutoff))
    except:
        print('simu not fitted')

    # repeat fot exp
    try:
        obj = obj
        p0 = [max(escorelist), np.median(eparamlist), 1, min(escorelist)]  # this is an mandatory initial guess
        popt, pcov = curve_fit(obj, eparamlist, escorelist, method='trf', p0=p0)
        print('coefs:', popt)
        r2 = np.round(r2_score(np.asarray(escorelist), obj(np.asarray(eparamlist), *popt)), 3)
        minum = np.round(np.min(eparamlist), 3)
        maximum = np.round(np.max(eparamlist), 3)
        func = lambda x: obj(x, *popt)
        cutoff = inversefunc(func, y_values=[scorelimit], domain=domain)[0]  ###############
        cutoff = round(cutoff, 3)
        ax.plot(np.linspace(minum, maximum, 100), obj(np.linspace(minum, maximum, 100), *popt), '--', c='r',
                label='R\N{SUPERSCRIPT TWO}: ' + str(r2))
        dot = ax.scatter(cutoff, scorelimit, c='r', marker='^')
        dot.set_label('\nexpCutoff = ' + str(cutoff))
    except:
        print(' ')
    ax.legend()

    plt.xlabel(xytitlenames[0], fontsize=16)
    plt.ylabel(xytitlenames[1], fontsize=16)
    plt.title(xytitlenames[2], fontsize=18)
    # labels = [' ','0.5', '1', '2', '4', '6', '8', '10', '12']
    # ax.set_xticklabels(labels)
    plt.grid('on')
    ax.tick_params(direction='in')

    ax.set_ylim(ylim)
    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    plt.show()


def visualscore_compare(paramlist, scorelist, xytitlenames, isExtropolation, order, scorelimit, eparamlist, escorelist,
                        ylim):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.canvas.draw()
    ax.scatter(paramlist, scorelist, c='g', label='Ziatdinov')
    ax.scatter(eparamlist, escorelist, c='b', label='new')
    ax.legend()
    if isExtropolation:
        # paramlist = paramlist[3:7]
        # scorelist = scorelist[3:7]
        z = np.polyfit(paramlist, scorelist, order)
        p = np.poly1d(z)
        print('z:', z)

        # res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t_train, y_train))

        r2 = np.round(r2_score(np.asarray(scorelist), p(np.asarray(paramlist))), 3)
        min = np.round(np.min(paramlist), 2)
        max = np.round(np.max(paramlist), 2)
        cutoff = solve(p(x) - scorelimit, x)[0]
        cutoff = round(cutoff, 3)
        ax.plot(np.linspace(min, max, 100), p(np.linspace(min, max, 100)), '--', c='r',
                label='R\N{SUPERSCRIPT TWO}: ' + str(r2))

        dot = ax.scatter(cutoff, scorelimit, c='r', marker='^')
        dot.set_label('\nCutoff = ' + str(cutoff))

        ax.legend()

    plt.xlabel(xytitlenames[0], fontsize=16)
    plt.ylabel(xytitlenames[1], fontsize=16)
    plt.title(xytitlenames[2], fontsize=18)
    # labels = [' ','0.5', '1', '2', '4', '6', '8', '10', '12']
    # ax.set_xticklabels(labels)
    plt.grid('on')
    ax.tick_params(direction='in')

    ax.set_ylim(ylim)
    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    plt.show()
