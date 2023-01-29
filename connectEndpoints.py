import scipy 
import skimage
import time
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import scipy.ndimage.filters
from scipy.ndimage.filters import gaussian_laplace
import math
from matplotlib.patches import Circle
import matplotlib as mat
import cv2
import numpy as np

from scipy import signal
from PIL import Image
from pylab import *

import json
import argparse

""" node_dicts : [{
        "id":i,
        "pred_class": all_pred_classes[i],
        "border": border,
        "center": ((pointUL + pointLR)/2).astype(int),
        "radius": int(np.linalg.norm(pointUL - pointLR)/2)
    }]
"""

class connectNodes:
    """
    input: 1. deshadowed, node masked, text masked image 
           2. node_dicts from node detection, format above

    usage: 
    node_map = connectNodes(node_dicts, masked_img.copy())
    node_map.get_adjacency_matrix()
    result is node_map.adjacency_matrix
    """

    # nodes are several endpoints generated from detected node's central and radius
    def __init__(self, node_dicts, masked_img):
        self.number_of_nodes = len(node_dicts)
        self.adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes))
        scale_percent = 500/masked_img.shape[1] # percent of original size
        width = int(masked_img.shape[1] * scale_percent)
        height = int(masked_img.shape[0] * scale_percent)
        dim = (width, height)
        resized = cv2.resize(masked_img, dim, interpolation = cv2.INTER_AREA)
        self.masked_img = resized
        self.eigen_vector = None
        self.skel_img = None
        self.eign_points = None
        self.img_eignValues = None
        self.scale_percent = scale_percent
        self.node_dicts = node_dicts.copy()
        for node in self.node_dicts:
            for i in range(4):
                node["border"][i]*= scale_percent
            for i in range(2):
                node["center"][i] = int(node["center"][i] * scale_percent)
            node["radius"] = int(node["radius"] * scale_percent)
            
    #--------------------------------------helper functions below-----------------------------------------------------
    def dist(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    
    # determines whether v or reverse v is in same direction as ori
    def revvecVec(self, ori, v): 
        cos_angle = (ori[0] * v[0] + ori[1] * v[1]) / np.sqrt((ori[0]**2 + ori[1]**2) * (v[0]**2 + v[1]**2))
        if cos_angle > 0:
            return v
        else:
            return -v
        
    def cos_angle(self, ori, v):
        cos_angle = (ori[0] * v[0] + ori[1] * v[1]) / np.sqrt((ori[0]**2 + ori[1]**2) * (v[0]**2 + v[1]**2))
        return cos_angle
    
    def pointInMask(self, point):
        for node in self.node_dicts:
            radius = node["radius"]+5
            center = [node["center"][1], node["center"][0]]
            if self.dist(center, point) < radius:
                return True, node
        return False, None
        
    
    def sparse_subset2(self, points, r):
        #from https://codereview.stackexchange.com/questions/196104/removing-neighbors-in-a-point-cloud
        #Return a maximal list of elements of points such that no pairs of points in the result have distance less than r.
        result = []
        for p in points:
            if all(dist(p, q) >= r for q in result):
                result.append(p)
        return np.array(result)
    
    def generate_circle(self, center, radius, xmax, ymax, number_of_samples):
        N = number_of_samples # number of discrete sample points to be generated along the circle
        circlePoints = np.array([])
        for k in range(N):
            # compute
            angle = math.pi*2*k/N
            dx = radius*math.cos(angle)
            dy = radius*math.sin(angle)
            circle_point = np.zeros(2)
            circle_point[0] = int(round(center[0] + dx))
            circle_point[1] = int(round(center[1] + dy))
            # add to list
            if circle_point[0] >= 0 and circle_point[1] >= 0 and circle_point[0] < xmax and circle_point[1] < ymax:
                circlePoints = np.append(circlePoints , circle_point)
        return circlePoints.reshape(-1, 2).astype(int)
    
    def get_list_of_pixels_from_list_of_coordinates(self, all_points_on_cirlce):
        result = []
        for point in all_points_on_cirlce:
            result.append(self.skel_img[point[1],point[0]])
        return np.array(result)
    
    def find_closest_node_from_point(self, point):
        min_dist = np.linalg.norm(self.node_dicts[0]["center"] - point)
        closest_node = self.node_dicts[0]
        for i, this_node in enumerate(self.node_dicts):
            this_dist = np.linalg.norm(this_node["center"] - point)
            if this_dist < min_dist:
                closest_node = this_node
                min_dist = this_dist
        if min_dist > 20: return None
        return closest_node
    
    def gauss_derivative_kernels(self, size, sizey=None):
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
    
    def gauss_kernel(self, size, sizey = None):
        """ Returns a normalized 2D gauss kernel array for convolutions """
        size = int(size)
        if not sizey:
            sizey = size
        else:
            sizey = int(sizey)
        x, y = mgrid[-size:size+1, -sizey:sizey+1]
        g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
        return g / g.sum()
    
    def in_node_range(self, x, y, orig):
        range_limit = 3
        for node in self.node_dicts:
            center = node["center"]
            radius = node["radius"]
            if center[1] == orig[0] and center[0] == orig[1]: continue
            if self.dist((x, y), (center[1], center[0])) < radius + range_limit:
                return node
        return None
    
    def get_all_points_on_bounding_box(self, xmin, xmax, ymin, ymax, threshold = 150):
        points = []
        delta = (-1, 0, 1)
        for y in range(ymin, ymax):
            for xlim in (xmin, xmax):
                for d in delta:
                    x = xlim + d
                    if x >= 0 and x < self.masked_img.shape[1] and y >= 0 and y < self.masked_img.shape[0]:
                        if self.skel_img[y, x] > threshold:
                            points.append((x, y))
                    
        for x in range(xmin, xmax):
            for ylim in (ymin, ymax):
                for d in delta:
                    y = ylim + d
                    if x >= 0 and x < self.masked_img.shape[1] and y >= 0 and y < self.masked_img.shape[0]:
                        if self.skel_img[y, x] > threshold:
                            points.append((x, y))
        points = np.asarray(list(set(points)))
        return points.reshape(-1, 2).astype(int)
    
    def get_points_on_circle(self, center, radius, xmax, ymax, number_of_samples, threshold = 150):
        N = number_of_samples
        circlePoints = []
        delta = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for k in range(N):
            # compute
            angle = math.pi*2*k/N
            dx = radius*math.cos(angle)
            dy = radius*math.sin(angle)
            circle_point = np.zeros(2)
            circle_point[0] = int(round(center[0] + dx))
            circle_point[1] = int(round(center[1] + dy))
            # add to list
            for mx, my in delta:
                point = (int(circle_point[0] + mx), int(circle_point[1] + my))
                if point[0] >= 0 and point[1] >= 0 and point[0] < xmax and point[1] < ymax:
                    if self.skel_img[point[0], point[1]] > threshold:
                        circlePoints.append(point)
        circlePoints = np.asarray(list(set(circlePoints)))
        return circlePoints.reshape(-1, 2).astype(int)
    
    def get_prominent_points(self, center, radius, border = None, delta = 5, mode='circle'):
        result = []
        if mode == 'circle':
        # Return a list of coordinates indicating the points on the lines
        
            radius += delta
            all_points_on_cirlce = self.generate_circle(center, radius, self.skel_img.shape[1], self.skel_img.shape[0], 1000)
            all_pixels = self.get_list_of_pixels_from_list_of_coordinates(all_points_on_cirlce)
            avg_color = all_pixels.mean(axis=0)
            
            # print(avg_color)
            for i, this_pixel in enumerate(all_pixels):
                if np.linalg.norm(this_pixel - avg_color) > 100:
                    result.append(all_points_on_cirlce[i])
            #result = self.sparse_subset2(result, 10)
            #result = np.asarray(result)
        elif mode == 'square':
            if border == None:
                print('border imformation needed for get_prominent_points()')
                return result
            xmin, xmax, ymin, ymax = int(border[0]-delta), int(border[2]+delta), int(border[1]-delta), int(border[3]+delta)
            points = self.get_all_points_on_bounding_box(xmin, xmax, ymin, ymax)
            result = np.asarray(points)
        return result
    
    #-----------------------------------core functions below-----------------------------------------------------------
    # call x-y reversed start_point, center
    def line_tracker(self, start_point, center):
        # track around the line and return the start_point and end_point of the line
        # each time use the current point, follow the eigenvalue direction, go [step] pixels
        # until the next closest point is greater than step
        
        # convet the coordinate back
        x, y = start_point[0], start_point[1]
        step = 5
        count = 0
        step_limit = max(int(self.masked_img.shape[0]), int(self.masked_img.shape[1]))//step
        v = self.img_eignValues[(x, y)]
        eival, eivec = v[0], v[1]
        if self.dist((x+eivec[0]*step, y+eivec[1]*step), center) > self.dist((x-eivec[0]*step, y-eivec[1]*step), center):
            cur_dir = eivec
        else:
            cur_dir = -eivec
        pre_dir = (start_point[0]-center[0], start_point[1] - center[1])
        while True:
            dx, dy = cur_dir[0], cur_dir[1]
            nx, ny = x + dx, y + dy
            if count > step_limit:
                print('step limit hit')
                break
            close_node = self.in_node_range(x, y, center)
            if close_node:
                return close_node
            # case where the next point is on the line
            if (nx, ny) in self.img_eignValues:
                v = self.img_eignValues[(nx, ny)]
                eival, eivec = v[0], v[1]
                next_dir = self.revvecVec((dx, dy), eivec) * step
            #case where the next point is out of the line, recalculate nx, ny
            else:
                all_points_on_circle = self.get_points_on_circle((x, y), step, self.skel_img.shape[0], self.skel_img.shape[1], 20)
                if len(all_points_on_circle) > 1:
                    cur_best = None
                    best_cos = 0
                    for point in all_points_on_circle:
                        v = (point[0]-x, point[1]-y)
                        cos = self.cos_angle((dx, dy), v)
                        cos_pre = self.cos_angle(pre_dir, v)
                        cos = max(cos, cos_pre)
                        if cos > best_cos:
                            best_cos = cos
                            next_bext = v
                    if best_cos > 0:
                        next_dir = next_bext
#                         nx, ny = x
                    else:
                        break
                else:
                     break
            x, y = nx, ny
            pre_dir = cur_dir
            cur_dir = next_dir
            count += 1
        return self.find_closest_node_from_point(np.asarray([y, x]))

            
    def skeletonize(self):
        img = self.masked_img
        img = 255 - img
        plt.figure(figsize = (10,10))
        # Threshold the image
        ret,img = cv2.threshold(img, 30, 255, 0)

        # Step 1: Create an empty skeleton
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)

        # Get a Cross Shaped Kernel
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

        # Repeat steps 2-4
        while True:
            #Step 2: Open the image
            open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            #Step 3: Substract open from the original image
            temp = cv2.subtract(img, open)
            #Step 4: Erode the original image and refine the skeleton
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()
            # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
            if cv2.countNonZero(img)==0:
                break
        self.skel_img = skel
        
        points = []
        for i in range(len(skel)):
            for j in range(len(skel[0])):
                if skel[i][j] != 0:
                #or self.masked_img[i][j] < 235:
                    points.append((i, j))
        self.eign_points = points
    
    def compute_eigenvalues_pixels_points(self):
        #derivatives
        gx,gy = self.gauss_derivative_kernels(5)
        imx = signal.convolve(self.skel_img,gx, mode='same')
        imy = signal.convolve(self.skel_img,gy, mode='same')

        #kernel for blurring
        gauss = self.gauss_kernel(5)
        #compute components of the structure tensor
        Wxx = signal.convolve(imx*imx,gauss, mode='same')
        Wxy = signal.convolve(imx*imy,gauss, mode='same')
        Wyy = signal.convolve(imy*imy,gauss, mode='same')
        Wdet = Wxx*Wyy - Wxy**2
        Wtr = Wxx + Wyy
        R = Wdet - 0.05 * Wtr ** 2
        n, m = self.skel_img.shape
        eigenvalues = {}
        for point in self.eign_points:
            i, j = point[0], point[1]
            cur = [[Wxx[i][j], Wxy[i][j]], 
                   [Wxy[i][j], Wyy[i][j]]]
            w, v = eig(cur)
            idx = np.argmax(w)
            if R[i][j] < -20000:
                eigenvalues[(i, j)] = (w[idx], v[idx])
            else: # for dubug purpose
                eigenvalues[(i, j)] = (w[idx], v[idx])
        self.img_eignValues = eigenvalues
    
    
    def get_adjacency_matrix(self):
        # store the result adjacency matrix, class main function
        for node in self.node_dicts:
            if node["pred_class"] == 0:
                prominent_points = self.get_prominent_points(node["center"], node["radius"],node['border'], delta = 3, mode = 'circle')
            else:
                prominent_points = self.get_prominent_points(node["center"], node["radius"],node['border'], delta = 3, mode = 'square')
            center = node["center"]
            for prominent_point in prominent_points:
                dest = self.line_tracker((prominent_point[1], prominent_point[0]), (center[1], center[0]))
                if dest and dest != node:
                    self.adjacency_matrix[node["id"], dest["id"]] = 1
                    self.adjacency_matrix[dest["id"], node["id"]] = 1
    # ---------------------------------test(debug)functions below-------------------------------------------------------------
    
    def show_masked_img(self):
        if self.masked_img.any():
            plt.figure(figsize = (10,10))
            plt.imshow(255-self.masked_img)
        
    def show_skel_img(self):
        if self.skel_img.any():
            plt.figure(figsize = (10,10))
            plt.imshow(self.skel_img,cmap='gray', vmin=0, vmax=255)
            
    def show_all_direction(self, color = (255, 255, 0)):
        image = self.skel_img.copy()
        n, m = image.shape
        count = 0
        for k in self.img_eignValues:
            count += 1
            if count % 10 == 0:
                v = self.img_eignValues[k]
                x, y = k[1], k[0]
                eival, eivec = v[0], v[1]
                cv2.arrowedLine(image, (x, y),  (int(x + eivec[1]* 20), int(y + eivec[0] * 20)), color = color,thickness = 1)
        plt.figure(figsize = (10,10))
        imshow(image,cmap='gray', vmin=0, vmax=255)
        
    def show_all_circles(self, points, color='r'):
        """
        image: numpy array, representing the grayscsale image
        cx, cy: numpy arrays or lists, centers of the detected blobs
        rad: numpy array or list, radius of the detected blobs
        """
        img = self.skel_img
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_aspect('equal')
        ax.imshow(img, cmap='gray')
        for x, y in points:
            circ = Circle((x, y), 5, color=color, fill=False)
            ax.add_patch(circ)
        plt.title('%i circles' % len(points))
        plt.show()
    
    def show_nodes_start_points(self, nodes, points, color = 'r'):
        img = self.skel_img
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_aspect('equal')
        ax.imshow(img, cmap='gray')
        for x, y in points:
            circ = Circle((x, y), 5, color=color, fill=False)
            ax.add_patch(circ)
        plt.title('%i circles' % len(points))
        for x, y in nodes:
            circ = Circle((x, y), 3, color='green', fill=True)
            ax.add_patch(circ)
        plt.show()
    
    def printNodeIdOnOriginalImage(self):
        img = self.masked_img.copy()
        for node in self.node_dicts:
            cv2.putText(img, text=str(node['id']), org=(node['center'][0], node['center'][1]), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(0, 255, 0),thickness=1)
        imshow(img, cmap='gray', vmin=0, vmax=255)
        # call x-y reversed start_point, center
        
    def line_tracker_test(self, start_point, center):
        # track around the line and return the start_point and end_point of the line
        # each time use the current point, follow the eigenvalue direction, go [step] pixels
        # until the next closest point is greater than step
        
        # convet the coordinate back
        img = self.skel_img
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_aspect('equal')
        ax.imshow(img, cmap='gray')
        x, y = start_point[0], start_point[1]
        circ = Circle((y, x), 2, color='g', fill=True)
        ax.add_patch(circ)
        
        step = 5
        count = 0
        step_limit = max(int(self.masked_img.shape[0]), int(self.masked_img.shape[1]))//step
        v = self.img_eignValues[(x, y)]
        eival, eivec = v[0], v[1]
        if self.dist((x+eivec[0]*step, y+eivec[1]*step), center) > self.dist((x-eivec[0]*step, y-eivec[1]*step), center):
            cur_dir = eivec
        else:
            cur_dir = -eivec
        pre_dir = (start_point[0]-center[0], start_point[1] - center[1])
        while True:
            dx, dy = cur_dir[0], cur_dir[1]
            nx, ny = x + dx, y + dy
            print(nx, ny)
            circ = Circle((y, x), 2, color='g', fill=True)
            ax.add_patch(circ)
            if count > step_limit:
                print('step limit hit')
                break
            close_node = self.in_node_range(x, y, center)
            if close_node:
                return close_node
            # case where the next point is on the line
            if (nx, ny) in self.img_eignValues:
#                 print("next point in line")
                v = self.img_eignValues[(nx, ny)]
                eival, eivec = v[0], v[1]
                next_dir = self.revvecVec((dx, dy), eivec) * step
            #case where the next point is out of the line, recalculate nx, ny
            else:
                all_points_on_circle = self.get_points_on_circle((x, y), step, self.skel_img.shape[0], self.skel_img.shape[1], 20)
#                 print('next point not in line')
                if len(all_points_on_circle) > 1:
                    cur_best = None
                    best_cos = 0
                    for point in all_points_on_circle:
                        v = (point[0]-x, point[1]-y)
                        cos = self.cos_angle((dx, dy), v)
                        cos_pre = self.cos_angle(pre_dir, v)
                        cos = max(cos, cos_pre)
                        if cos > best_cos:
                            best_cos = cos
                            next_bext = v
                    if best_cos < 0.9:
                        print("best cos", best_cos)
                    if best_cos > 0:
                        next_dir = next_bext
#                         nx, ny = x
                    else:
                        break
                else:
                     break
            x, y = nx, ny
            pre_dir = cur_dir
            cur_dir = next_dir
            count += 1
        circ = Circle((y, x), 5, color='r', fill=False)
        ax.add_patch(circ)
        plt.figure(figsize = (10,10))
        plt.show()
        return self.find_closest_node_from_point(np.asarray([y, x]))

    def get_adjacency_matrix_test(self):
        # store the result adjacency matrix, class main function
        G = {}
        for node in self.node_dicts:
#             if node["id"] != 7: continue
            if node["pred_class"] == 0:
                prominent_points = self.get_prominent_points(node["center"], node["radius"],node['border'], delta = 3, mode = 'circle')
            else:
                prominent_points = self.get_prominent_points(node["center"], node["radius"],node['border'], delta = 3, mode = 'square')
            #print(node["center"], node["radius"]+10)
            #print(prominent_points)
            #self.show_nodes_start_points([node["center"]], prominent_points)
            center = node["center"]
            for prominent_point in prominent_points:
                dest = self.line_tracker_test((prominent_point[1], prominent_point[0]), (center[1], center[0]))
                if dest and dest != node:
                    if node['id'] not in G:
                        G[node['id']] = set()
                    G[node['id']].add(dest['id'])
                    self.adjacency_matrix[node["id"], dest["id"]] = 1
                    self.adjacency_matrix[dest["id"], node["id"]] = 1
        for s in G:
            print(s, '->', G[s])

