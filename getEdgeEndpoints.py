import numpy as np
import skimage
from skimage.transform import rescale, resize
from skimage.color import rgb2gray
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib as mat
import scipy 
import scipy.ndimage.filters
from scipy.ndimage.filters import gaussian_laplace
from scipy import signal
import cv2
import math
from PIL import Image
from pylab import *

# helper function that is used to show the direction of the engienvectors of some points
def show_all_direction(image,eignvalues, color = (255, 255, 0)):
    print(len(eignvalues))
    n, m = image.shape
    for k in eignvalues:
        v = eignvalues[k]
        x, y = k[1], k[0]
        eival, eivec = v[0], v[1]
        cv2.arrowedLine(image, (x, y),  (int(x + eivec[1]* 20), int(y + eivec[0] * 20)), color = color,thickness = 1)
    plt.figure(figsize = (10,10))
    imshow(image)

# helper function that is used for showing all circles on the image
def show_all_circles(image, cx, cy, rad, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)
    plt.title('%i circles' % len(cx))
    plt.show()

# helper function that is used to show the endpoints of the image
def show_points(ax, points, r = 3, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    ax.set_aspect('equal')
    i = 0
    if len(points) < 2:
        pass
    for x, y in points:
        if i % 2 == 0:
            color = 'r'
        else:
            color = 'b'
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)
        i += 1

# the following few fucntions performs blocb detections on the image

# filtering the image (two implmementations)
# one that increases filter size, and one that downsamples the image
# For timing, use time.time()
def filter_up_kernel(image, N, sigma):
    h = image.shape[0]
    w = image.shape[1]
    scale_space = np.empty((N,h,w))
    k = 1.25
    for i in range(N):
        cur_filtered = gaussian_laplace(image, sigma = sigma, truncate = 2)
        cur_filtered = (sigma**2 * cur_filtered)**2
        scale_space[i] = cur_filtered
        sigma *= k
    return scale_space

def non_maximum_supression_faster(sigma_space, kernel_width):
    max_sigma_space = np.empty(sigma_space.shape)
    for i in range(len(sigma_space)):
        max_sigma_space[i] = scipy.ndimage.rank_filter(sigma_space[i], rank = -1, size=(kernel_width, kernel_width))
    def kernel(in_array):
#         print("before", in_array)
        median = in_array[len(in_array)//2]
        if(median == max(in_array)):
            return median
        return 0
    NMS_space = scipy.ndimage.generic_filter(max_sigma_space, kernel, footprint=np.ones((kernel_width, 1, 1)))
    for i in range(len(sigma_space)):
        for row in range(sigma_space.shape[1]):
            for col in range(sigma_space.shape[2]):
                if NMS_space[i, row, col] == 0:
                    continue
                if NMS_space[i, row, col] > sigma_space[i, row, col]:
                    NMS_space[i, row, col] = 0
    return NMS_space

# main function to be called for the blob detection
def blob_dection_up(grayscale_erode, sigma, thre, kernel_width):
    up_filtered = filter_up_kernel(grayscale_erode, 3, sigma)
    nms_up = non_maximum_supression_faster(up_filtered, kernel_width)
    points = np.where(nms_up > thre)
    k = 1.25    
    return points

# the following few fucntions are used to get the image derivatives of the detected blobs

def gauss_derivative_kernels(size, sizey=None):
    """ returns x and y derivatives of a 2D 
        gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    y, x = mgrid[-size:size+1, -sizey:sizey+1]
    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    gy = - y * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    return gx,gy

def gauss_kernel(size, sizey = None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def compute_eigenvalues_pixels_points(im, points):
    #derivatives
    gx,gy = gauss_derivative_kernels(3)
    imx = signal.convolve(im,gx, mode='same')
    imy = signal.convolve(im,gy, mode='same')
    
    #kernel for blurring
    gauss = gauss_kernel(3)
    #compute components of the structure tensor
    Wxx = signal.convolve(imx*imx,gauss, mode='same')
    Wxy = signal.convolve(imx*imy,gauss, mode='same')
    Wyy = signal.convolve(imy*imy,gauss, mode='same')
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    R = Wdet - 0.05 * Wtr ** 2
    n, m = im.shape
    eigenvalues = {}
    for point in points:
        i, j = point[0], point[1]
        cur = [[Wxx[i][j], Wxy[i][j]], 
               [Wxy[i][j], Wyy[i][j]]]
        w, v = eig(cur)
        idx = np.argmax(w)
#         print('E-value:', w)
#         print('E-vector', v)
        eigenvalues[(i, j)] = (w[idx], v[idx])
        for n, m in ((int(i+v[idx][0] * 15), int(j+v[idx][1] * 15)), (int(i-v[idx][0] * 15), int(j-v[idx][1] * 15))):
            if n > 0 and m > 0 and n < im.shape[0] and m < im.shape[1] and R[n][m] < -100000:
                cur = [[Wxx[n][m], Wxy[n][m]], 
                       [Wxy[n][m], Wyy[n][m]]]
                w, v = eig(cur)
                idx = np.argmax(w)
                eigenvalues[(n, m)] = (w[idx], v[idx])
    return eigenvalues

def connect_blobs(eigenvalues, n, m, image):
    feature_vec = set()
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    res = []
    for k in eigenvalues:
        i, j = k[1], k[0]
        v = eigenvalues[k][1]
        x, y = v[1], v[0]
        feature_vec.add((i, j, x, y))
    tmp_res = []
    q = []
    step = 0
    while feature_vec:
        if len(q) == 0:
            if len(tmp_res) == 2 and step > 5:
                show_points(ax, tmp_res, r = 5, color = 'red')
                res.append(tmp_res)
            tmp_res = []
            cur = feature_vec.pop()
            q = [((cur[0], cur[1]), (cur[2], cur[3])), ((cur[0], cur[1]), (-cur[2], -cur[3]))]
            step = 0
        cord, dire = q.pop()
        i, j, x, y = cord[0], cord[1], dire[0], dire[1]
        
        # filter out boudary points
        if i < 10 or i > m - 10 or j < 10 or j > n - 10:
            continue
        best_next = None
        # find best match within feature_vec
        v1 = (x, y)
        for vec in feature_vec:
            ti, tj, tx, ty = vec[0], vec[1], vec[2], vec[3]
            di = ti - i
            dj = tj - j
            dist = np.sqrt(dj**2 + di**2)
            if dist < 30:
                v2 = (tx, ty)
                cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / np.sqrt((v1[0]**2 + v1[1]**2) * (v2[0]**2 + v2[1]**2))
                if cos_angle > 0.9:
                    if best_next == None or (best_next[2] > dist and cos_angle > 0.96):
                        best_next = ((ti, tj), (tx, ty), dist)
                elif cos_angle < -0.9:
                    if best_next == None or (best_next[2] > dist and cos_angle < -0.96):
                        best_next = ((ti, tj), (-tx, -ty), dist)
        if best_next != None:
            # print(best_next)
            feature_vec.discard((best_next[0][0], best_next[0][1], best_next[1][0], best_next[1][1]))
            feature_vec.discard((best_next[0][0], best_next[0][1], -best_next[1][0], -best_next[1][1]))
            q.append((best_next[0], best_next[1]))
            step += 1
        else:
            tmp_res.append((i, j))
    plt.savefig('edges.jpg')
    return res

def myrgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def getEdgeEndpoints(img):
    cv2.imwrite('image.png',img)

    start_time = time.time()
    # change the image to its binary inversion
    grayscale = rgb2gray(img)
    grayscale_erode = 1- cv2.erode(grayscale, np.ones((5,5), np.uint8), iterations=1)
    # blob_detection to get point location
    sigma = 2
    thre = 0.01
    k = 1.25
    points = blob_dection_up(grayscale_erode, sigma, thre, 5)
    point_array = np.concatenate((points[1].reshape(-1, 1), points[2].reshape(-1, 1)), axis = 1)
    # show_all_circles(grayscale, points[2], points[1], sigma*k**points[0] * np.sqrt(2))
    # using 2nd moment matrix to get points direction as well as expanding those points
    grayscale = np.double(myrgb2gray(img))
    grayscale_erode = 1 - cv2.erode(grayscale, np.ones((3,3), np.uint8), iterations=1)
    ev = compute_eigenvalues_pixels_points(grayscale_erode, point_array)
    show_all_direction(grayscale_erode, ev)
    # get resulting image as well as showing it
    res = connect_blobs(ev, img.shape[0], img.shape[1], img)
    process_time = time.time() - start_time
    print("process time: ", process_time)
    return res
