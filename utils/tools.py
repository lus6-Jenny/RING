# utils for dataset processing

import os
import cv2
import copy
import errno
import torch
import pickle
import numpy as np
import open3d as o3d
import utils.config as cfg
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import torchvision.transforms.functional as fn
from sklearn.neighbors import NearestNeighbors, KDTree

# Coordinates of test region centers
# nclt_test_region_centers = np.array([[-50,-250],[-50,150],[150,-250],[150,150]])
nclt_test_region_centers = np.array([[50, -50]])
mulran_test_region_centers = np.array([[345090.0743, 4037591.323], [345090.483, 4044700.04],
                                [350552.0308, 4041000.71], [349252.0308, 4044800.71]])
kitti_test_region_centers = np.array([[50, -50]])

oxford_radar_test_region_centers = np.array([[5735500, 620000], [5735500, 620500], [5735000, 620500], [5735000, 620000]])

# Radius of the test region
nclt_test_region_radius = 220
mulran_test_region_radius = 500
kitti_test_region_radius = 100
oxford_radar_test_region_radius = 500

# Boundary between the train and test region
test_region_boundary = 50


# Make directory
def mkdir(dirname):
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


# Load pickle file
def load_pickle(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        print("Pickle Data Loaded.")
        return data


# Find the closest timestamp of the target point cloud
def find_nearest_ndx(ts, timestamps):
    ndx = np.searchsorted(timestamps, ts)
    if ndx == 0:
        return ndx
    elif ndx == len(timestamps):
        return ndx - 1
    else:
        assert timestamps[ndx-1] <= ts <= timestamps[ndx]
        if ts - timestamps[ndx-1] < timestamps[ndx] - ts:
            return ndx - 1
        else:
            return ndx


# Sample point clouds at equidistant intervals
def sample_at_intervals(northing, easting, prev_northing, prev_easting, sampling_gap):
    is_submap = False
    euclidean = np.abs(np.sqrt((prev_northing-northing)**2 + (prev_easting-easting)**2))
    if euclidean >= sampling_gap:
        is_submap = True
    return is_submap


# Check if the location is in the train set
def check_in_train_set(pos, dataset):
    # returns true if pos is in train split
    assert pos.ndim == 2
    assert pos.shape[1] == 2    
    if dataset == "nclt":
        test_region_centers = nclt_test_region_centers
        test_region_radius = nclt_test_region_radius
    elif dataset == "mulran":
        test_region_centers = mulran_test_region_centers
        test_region_radius = mulran_test_region_radius
    elif dataset == "kitti":
        test_region_centers = kitti_test_region_centers
        test_region_radius = kitti_test_region_radius
    elif dataset == "oxford_radar":
        test_region_centers = oxford_radar_test_region_centers
        test_region_radius = oxford_radar_test_region_radius
    dist = distance_matrix(pos, test_region_centers)
    mask = (dist > test_region_radius + test_region_boundary).all(axis=1)
    return mask


# Check if the location is in the test set
def check_in_test_set(pos, dataset):
    # returns true if position is in evaluation split
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    if dataset == "nclt":
        test_region_centers = nclt_test_region_centers
        test_region_radius = nclt_test_region_radius
    elif dataset == "mulran":
        test_region_centers = mulran_test_region_centers
        test_region_radius = mulran_test_region_radius 
    elif dataset == "kitti":
        test_region_centers = kitti_test_region_centers
        test_region_radius = kitti_test_region_radius        
    elif dataset == "oxford_radar":
        test_region_centers = oxford_radar_test_region_centers
        test_region_radius = oxford_radar_test_region_radius           
    dist = distance_matrix(pos, test_region_centers)
    mask = (dist < test_region_radius).any(axis=1)
    return mask


# Check if it's a new place (class) in a trajectory of train dataset
def check_train_class(pose, prev_poses, threshold=cfg.train_sampling_gap):
    assert pose.ndim == 2
    assert pose.shape[1] == 2
    dist = distance_matrix(pose, prev_poses)
    is_new_class = (dist >= threshold).all(axis=1)
    # print("is_new_class_train", is_new_class)
    return is_new_class


# Check if it's a new place (class) in a trajectory of test dataset
def check_test_class(pose, prev_poses, threshold=cfg.test_sampling_gap):
    assert pose.ndim == 2
    assert pose.shape[1] == 2
    dist = distance_matrix(pose, prev_poses)
    is_new_class = (dist >= threshold).all(axis=1)
    # print("is_new_class_test", is_new_class)
    return is_new_class


# Calculate the distance between two poses
def calculate_dist(pose1, pose2):
    dist = np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)
    return dist


# Check if there is a loop (revisitness) in the map
def is_revisited(query_pose, map_poses, revisit_threshold):
    tree = KDTree(map_poses)
    # get the nearest neighbor
    dist, idx = tree.query(np.array([query_pose]), k=1)
    # dist, idx = tree.query(query_pose.reshape(1, -1) , k=1)
    if dist[0] < revisit_threshold:
        revisited = True
    else:
        revisited = False
    
    return revisited, dist, idx


# Convert robot id and index to key for multi-robot optimization
def robotid_to_key(robotid):
    char_a = 97
    keyBits = 64
    chrBits = 8
    indexBits = keyBits - chrBits
    outkey = char_a + robotid
    print("robotid: ", robotid, " outkey: ",outkey)
    return outkey << indexBits


def imshow(tensor, title=None):
    # print(tensor.type())
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    plt.imshow(image, cmap='jet')
    # plt.colorbar()
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig(title, bbox_inches='tight', pad_inches=0)
    plt.show()


# Plot trajectory of both map and query
def plot_trajectory(positions, split, save_path):
    positions = np.array(positions)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(positions[:, 1], positions[:, 0], 'b-', label=split)
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.title('Trajectory')
    plt.savefig(save_path)
    # plt.show()
    plt.close()