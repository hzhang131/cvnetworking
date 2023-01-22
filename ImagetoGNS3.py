from distutils.command.config import config
from webbrowser import open_new
import cv2
import numpy as np
import json as js
from scipy import ndimage
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo

import numpy as np
import scipy 
import skimage
import time
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import scipy.ndimage.filters
from scipy.ndimage.filters import gaussian_laplace
import cv2
import math
from matplotlib.patches import Circle
import matplotlib as mat

import argparse

import getEdgeEndpoints

import InfotoGNS3

# preprocessing on input image
def preprocessing_img(img): 
  rgb_planes = cv2.split(img)

  result_planes = []
  result_norm_planes = []
  for plane in rgb_planes:
      dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
      bg_img = cv2.medianBlur(dilated_img, 21)
      diff_img = 255 - cv2.absdiff(plane, bg_img)
      norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      result_planes.append(diff_img)
      result_norm_planes.append(norm_img)

  result = cv2.merge(result_planes)
  result_norm = cv2.merge(result_norm_planes)
  return result

# This version uses COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
def get_known_icons_from_image(img:np.ndarray, model_path:str):
  """
    Description:
      Take img and return the masked img without the icons and a list of dicts of nodes 
    Parameters:
      img (np.ndarray) : The img containing the drawing.
      model_path (str) : The path to the model_final.pth file of 1.train_get_KOWN_icons.
    Return:
      1.Masked img without the icons
      2.A list of dicts of nodes 
  """
  cfg = get_cfg()

  # config of THIS model 
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

  # number of classes 
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

  # path for final model
  cfg.MODEL.WEIGHTS = model_path

  # test threshhold
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

  predictor = DefaultPredictor(cfg) # get predictor

  outputs = predictor(img)
  # use mask in output to get that segment in image
  # first, copy the original img
  masked_img = img.copy()
  # second, get the mask (mask size is img.shap*2, so shrink the mask.) and construct the node dicts along the way
  output_cpu = outputs["instances"].to("cpu")
  all_pred_classes = output_cpu.pred_classes.numpy()
  all_masks_numpy = output_cpu.pred_masks.numpy()

  mask = all_masks_numpy[0]
  node_dicts = []
  for i in range(len(all_pred_classes)):
    mask = np.logical_or(all_masks_numpy[i], mask)
    border = output_cpu.pred_boxes[i].tensor.numpy()[0]
    pointUL = border[0:2].astype(int)
    pointLR = border[2:4].astype(int)
    node_dicts.append({
        "id":i,
        "pred_class": all_pred_classes[i],
        "border": border,
        "center": ((pointUL + pointLR)/2).astype(int),
        "radius": int(np.linalg.norm(pointUL - pointLR)/2)
    })
  # get mask as numpy array
  mask = mask.astype(float)# transform mask array to an image
  mask = cv2.resize(mask, dsize=(masked_img.shape[1], masked_img.shape[0]), interpolation=cv2.INTER_CUBIC)# resize mask: shrink to masked_img's size use area average
  mask = mask.astype(bool)# back to boolean

  # dilate the mask
  mask = ndimage.binary_dilation(mask, [[True, True, True], 
                  [True, True, True], 
                  [True, True, True]])

  masked_img[mask] = 255.0 # mask input img with the mask from model, with white color
    
  return masked_img, node_dicts


# return the distance between two points
def dist(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def strengthen_binary(img):
  result = img.copy()
  avg_color = img.mean(axis=0).mean(axis=0)
  for i in range(result.shape[0]):
    for j in range(result.shape[1]):
      if np.linalg.norm(img[i,j]-avg_color) < 100:
        result[i,j] = [255,255,255]
      else:
        result[i,j] = [0,0,0]
  return result

def adjust_pix_color(x, y, image, threshold, color_avg):
    mean_difference = (abs(float(image[x][y][0]) - color_avg[0]) + abs(float(image[x][y][1]) - color_avg[1]) + abs(float(image[x][y][2]) - color_avg[2]))/3
    if  mean_difference  <= threshold:
        return True
    return False

def update_avg_color(x, y, image, color_avg):
    count = color_avg[3]
    color_avg[0] = (image[x][y][0] + color_avg[0] * count)/(count + 1)
    color_avg[1] = (image[x][y][1] + color_avg[1] * count)/(count + 1)
    color_avg[2] = (image[x][y][2] + color_avg[2] * count)/(count + 1)
    color_avg[3] += 1
    
def generate_circle(point, radius, xmax, ymax, number_of_samples):
# parameters
    N = number_of_samples # number of discrete sample points to be generated along the circle
    circlePoints = np.array([])
    for k in range(N):
        # compute
        angle = math.pi*2*k/N
        dx = radius*math.cos(angle)
        dy = radius*math.sin(angle)
        circle_point = np.zeros(2)
        circle_point[0] = int(round(point[0] + dx))
        circle_point[1] = int(round(point[1] + dy))
        # add to list
        if circle_point[0]  >= 0 and circle_point[1] >= 0 and circle_point[0] < xmax and circle_point[1] < ymax:
            circlePoints = np.append(circlePoints , circle_point)
    return circlePoints.reshape(-1, 2).astype(int)

def get_next_estimate(previous, current):
  delta_x = current[0] - previous[0]
  delta_y = current[1] - previous[1]
  norm = np.linalg.norm([delta_x, delta_y])
  delta_x_norm = delta_x/norm
  delta_y_norm = delta_y/norm
  return (current[0] + delta_x_norm, current[1] + delta_y_norm)

"""
This algorithm is used to track the line given the start point (or not necessarily the start point)
the return value of this function should be two points (start and end) with their final directions
"""
def line_tracker(point, image, image_copy):
    x1, y1 = point[0], point[1]
    if image_copy[x1][y1][0] == 255 or image_copy[x1][y1][2] == 255:
        return np.zeros(2), np.zeros(2)
    color_avg = [image[x1][y1][0], image[x1][y1][1], image[x1][y1][2], 1]
    circle_radius = 8
    color_threshold = 80
    # get a circle around one point with radius r
    points_on_circle = generate_circle(point, circle_radius, image.shape[0], image.shape[1], 200)
    initial_candidate = np.zeros((2,2))
    Flag = True
    count = 0
    # first loop that records the initial directions along the point 
    for circle_point in points_on_circle:
        if adjust_pix_color(circle_point[0], circle_point[1], image, color_threshold, color_avg) and Flag == True:
            initial_candidate[0] = circle_point
            update_avg_color(circle_point[0], circle_point[1], image, color_avg)
            Flag = False
        elif adjust_pix_color(circle_point[0], circle_point[1], image, color_threshold, color_avg) and Flag == False:
            if dist(initial_candidate[0], circle_point) > circle_radius * 1.2:
                initial_candidate[1] = circle_point
                update_avg_color(circle_point[0], circle_point[1], image, color_avg)
                break
    predecessor_0, predecessor_1 = point, point
    candidate_0 = initial_candidate[0]
    candidate_1 = initial_candidate[1]
    next_estimate_0 = get_next_estimate(point, candidate_0)
    next_estimate_1 = get_next_estimate(point, candidate_1)
    # visualize
    point_o = (point[1], point[0])
    if np.sum(initial_candidate[0]) > 0:
        point0 = (int(initial_candidate[0][1]), int(initial_candidate[0][0]))
        cv2.arrowedLine(image_copy, point_o, point0, (255, 0, 0), 1)
    if np.sum(initial_candidate[1]) > 0:
        point0 = (int(initial_candidate[1][1]), int(initial_candidate[1][0]))
        cv2.arrowedLine(image_copy, point_o, point0, (0, 0, 255), 1)
    # now we have initial candidate filled and ready to perform the main loop
    while np.sum(candidate_0) > 0 and count < 1000:
        # go direction 1:
        current_circle = generate_circle(candidate_0, circle_radius, image.shape[0], image.shape[1], 200)
        best_candidate = np.zeros(2)
        min_dist = 5000
        for circle_point in current_circle:
            if adjust_pix_color(circle_point[0], circle_point[1], image, color_threshold, color_avg):
                sanity_dist = dist(circle_point, predecessor_0)
                if sanity_dist < circle_radius * 1.2:
                    continue
                cur_dist = dist(circle_point, next_estimate_0)
                if cur_dist < min_dist:
                    cur_distance_o = cur_dist
                    best_candidate = circle_point
                    min_dist = cur_dist
        if np.sum(best_candidate) > 0:
            point_o = (int(candidate_0[1]), int(candidate_0[0]))
            point0 = (best_candidate[1], best_candidate[0])
            cv2.arrowedLine(image_copy, point_o, point0, (255, 0, 0), 1)
            next_estimate_0 = get_next_estimate(candidate_0, best_candidate)
        predecessor_0 = candidate_0
        candidate_0 = best_candidate
        if candidate_0[0] < 5 or candidate_0[1] < 5:
            break
        count += 1
    while np.sum(candidate_1) > 0 and count < 2000:
        # go direction 2:
        current_circle = generate_circle(candidate_1, circle_radius, image.shape[0], image.shape[1], 200)
        best_candidate = np.zeros(2)
        min_dist = 5000
        for circle_point in current_circle:
            if adjust_pix_color(circle_point[0], circle_point[1], image, color_threshold, color_avg):
                sanity_dist = dist(circle_point, predecessor_1)
                if sanity_dist < circle_radius * 1.2:
                    continue
                cur_dist = dist(circle_point, next_estimate_1)
                if cur_dist < min_dist:
                    cur_distance_o = cur_dist
                    best_candidate = circle_point
                    min_dist = cur_dist
        if np.sum(best_candidate) > 0:
            point_o = (int(candidate_1[1]), int(candidate_1[0]))
            point0 = (best_candidate[1], best_candidate[0])
            cv2.arrowedLine(image_copy, point_o, point0, (0, 0, 255), 1)
            next_estimate_1 = get_next_estimate(candidate_1, best_candidate)
        predecessor_1 = candidate_1
        candidate_1 = best_candidate
        if candidate_1[0] < 5 or candidate_1[1] < 5:
            break
        count += 1
    # we need to draw the lines along the 
    # cv2.circle(image_copy, predecessor_0[::-1].astype(int), 5, color = (255, 0, 0), thickness = 1)
    # cv2.circle(image_copy, predecessor_1[::-1].astype(int), 5, color = (0, 0, 255), thickness = 1)
    return predecessor_0, predecessor_1

def sparse_subset2(points, r):#from https://codereview.stackexchange.com/questions/196104/removing-neighbors-in-a-point-cloud
    """Return a maximal list of elements of points such that no pairs of
    points in the result have distance less than r.

    """
    result = []
    for p in points:
        if all(dist(p, q) >= r for q in result):
            result.append(p)
    return np.array(result)

def get_list_of_pixels_from_list_of_coordinates(img, list_of_coordinates):
  result = []
  for point in list_of_coordinates:
    result.append(
        img[point[1],point[0]]
    )
  return np.array(result)


def get_prominent_points(img, center, radius):
  """
  Return a list of coordinates indicating the points on the lines
  """
  all_points_on_cirlce = generate_circle(center, radius, img.shape[1], img.shape[0], 1000)
  all_pixels = get_list_of_pixels_from_list_of_coordinates(img, all_points_on_cirlce)
  avg_color = all_pixels.mean(axis=0)
  result = []
  for i, this_pixel in enumerate(all_pixels):
    if np.linalg.norm(this_pixel - avg_color) > 100:
      result.append(all_points_on_cirlce[i])
  result = sparse_subset2(result, 10)
  return result


def find_closest_node_from_point(node_dicts, point):

  min_dist = np.linalg.norm(node_dicts[0]["center"] - point)
  closest_node = node_dicts[0]
  for i, this_node in enumerate(node_dicts):
    this_dist = np.linalg.norm(this_node["center"] - point)
    if this_dist < min_dist:
      closest_node = this_node
      min_dist = this_dist
  if min_dist > 130:
    return None
  return closest_node


def get_adjacency_matrix(masked_img, node_dicts):
  number_of_nodes = len(node_dicts)
  adjacency_matrix = np.zeros((number_of_nodes, number_of_nodes))

  for node in node_dicts:
    prominent_points = get_prominent_points(masked_img, node["center"], node["radius"]+40)
    for prominent_point in prominent_points:
      start_point, end_point = line_tracker([prominent_point[1],prominent_point[0]], masked_img, masked_img.copy())
      start_node = find_closest_node_from_point(node_dicts, np.flip(start_point))
      end_node = find_closest_node_from_point(node_dicts, np.flip(end_point))
      adjacency_matrix[end_node["id"], start_node["id"]] = 1
      adjacency_matrix[start_node["id"], end_node["id"]] = 1
  return adjacency_matrix

def get_adjacency_matrix_from_blobs(masked_img, node_dicts):
  number_of_nodes = len(node_dicts)
  adjacency_matrix = np.zeros((number_of_nodes, number_of_nodes))
  edges = getEdgeEndpoints.getEdgeEndpoints(masked_img)
  for edge in edges:
    start_node = find_closest_node_from_point(node_dicts, edge[0])
    end_node = find_closest_node_from_point(node_dicts, edge[1])
    if start_node and end_node:
      adjacency_matrix[end_node["id"], start_node["id"]] = 1
      adjacency_matrix[start_node["id"], end_node["id"]] = 1

  return adjacency_matrix
    
def parse():
    """
    This function parses the command-line flags

    Parameters: 
      None
    Returns:
      parser.parse_args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', dest="img", type=str,
                        help='Input Free-hand sketch image path')
    parser.add_argument('--mat', dest="mat", type=str,
                        help='Predefined adjacency matrix path')
    parser.add_argument('--lis', dest="lis", type=str,
                        help='Predifined nodelist path')
    parser.add_argument('--gns3_file', dest="gns3_file", type=str,
                        help='gns3 topology description file')
    parser.add_argument('--name', dest="name", type=str,
                        help='GNS3 project name')
    parser.add_argument('--dir', dest="dir", type=str,
                        help='GNS3 project dir')
    parser.add_argument('--model', dest="model", type=str,
                        help='Recognition model path')
    parser.add_argument('--additional',dest="additional", type=str,
                          default="", help="additional arguments")

    return parser.parse_args()


def main():
    args = parse()
    user_img_path = args.img 
    model_final_path = "./model_final.pth" if not args.model else args.model
    name = args.name
    additional = args.additional.split(' ')
    outputDir = args.dir

    mat = args.mat
    list = args.lis
    gns3_file = args.gns3_file
    if mat != None:
      # with open(mat) as m:
      with open(mat) as m:
        adjacency_matrix = np.loadtxt(m)
      with open(list) as l:
        node_dicts = js.load(l)
      InfotoGNS3.generate_gns3file(name, outputDir, node_dicts, adjacency_matrix)
      configrator = InfotoGNS3.Configurator(outputDir+"/"+name+".gns3", outputDir, additional)
      configrator.configure_vpcs()
      configrator.configure_routers()
    elif gns3_file != None:
      # input GNS3 topology file into configurator.
      configrator = InfotoGNS3.Configurator(gns3_file, outputDir, additional)
      configrator.configure_vpcs()
      configrator.configure_routers()
    else:
      img = cv2.imread(user_img_path)
      scale_percent = int(1080/img.shape[1] * 100)
      width = int(img.shape[1] * scale_percent / 100)
      height = int(img.shape[0] * scale_percent / 100)
      dim = (width, height)
      # resize image
      img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
      processed_img = preprocessing_img(img.copy())
      masked_img, node_dicts = get_known_icons_from_image(processed_img.copy(), model_final_path)
      adjacency_matrix = get_adjacency_matrix_from_blobs(masked_img.copy(), node_dicts)
      print(type(adjacency_matrix))
      print(adjacency_matrix)
      print(type(node_dicts))
      print(node_dicts)
      print(type(node_dicts[0]))
      InfotoGNS3.generate_gns3file(name, outputDir, node_dicts, adjacency_matrix)
      configrator = InfotoGNS3.Configurator(outputDir+"/"+name+".gns3", outputDir, additional)
      configrator.configure_vpcs()
      configrator.configure_routers()
if __name__ == "__main__":
    main()