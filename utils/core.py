import os
import sys
import cv2
import time
import torch

import voxelocc
import voxelfeat
import numpy as np
import open3d as o3d
import utils.config as cfg
from utils.tools import imshow
import utils.vox_utils.vox as vox
from utils.circorr2 import circorr2
from utils.extract_local_descriptor import build_neighbors_NN

import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torch_radon import Radon, ParallelBeam, RadonFanbeam
from skimage.transform import radon, iradon, rescale

np.seterr(divide='ignore',invalid='ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
InstanceNorm = nn.InstanceNorm2d(1, affine=False, track_running_stats=False)


############ FFT Operations ############
# Apply 2D fourier transform to the input data
def forward_fft(input):
    median_output = torch.fft.fft2(input, dim=(-2, -1), norm="ortho")
    median_output_r = median_output.real
    median_output_i = median_output.imag
    output = torch.sqrt(median_output_r ** 2 + median_output_i ** 2)
    # output = torch.fft.fftshift(output)
    return output, median_output


# Apply 1D fourier transform to the row of input data
def forward_row_fft(input):
    median_output = torch.fft.fft2(input, dim=-1, norm="ortho")
    median_output_r = median_output.real
    median_output_i = median_output.imag
    output = torch.sqrt(median_output_r ** 2 + median_output_i ** 2)
    return output, median_output


# Apply 1D fourier transform to the column of input data
def forward_column_fft(input):
    median_output = torch.fft.fft2(input, dim=-2, norm="ortho")
    median_output_r = median_output.real
    median_output_i = median_output.imag
    output = torch.sqrt(median_output_r ** 2 + median_output_i ** 2)
    return output, median_output
############ FFT Operations ############


############ BEV Features Generation ############
# Generate BEV encoded by occupancy information
def generate_bev(pc):
    '''
    pc: original point cloud
    '''
    # returns cfg.num_height * cfg.num_ring * cfg.num_sector matrix 
    pc = pc[:,:3]
    bounds = (cfg.point_cloud["x_bound"][0], cfg.point_cloud["x_bound"][1], cfg.point_cloud["y_bound"][0], \
              cfg.point_cloud["y_bound"][1], cfg.point_cloud["z_bound"][0], cfg.point_cloud["z_bound"][1])
    scene_centroid = [0.0, 0.0, 0.0]
    scene_centroid = torch.from_numpy(np.array(scene_centroid).reshape([1, 3])).float()
    vox_util = vox.Vox_util(cfg.num_height, cfg.num_ring, cfg.num_sector,
                            scene_centroid=scene_centroid.cuda(),
                            bounds=bounds,
                            assert_cube=False)
    pc = torch.from_numpy(pc).unsqueeze(0)
    occ_mem = vox_util.voxelize_xyz(pc, cfg.num_height, cfg.num_ring, cfg.num_sector, assert_cube=False)
    pc_bev = occ_mem.permute(0,1,3,2,4).squeeze()
    dim = pc_bev.ndim
    if dim == 2:
        return pc_bev.permute(1,0).unsqueeze(0).to(device)
    elif dim == 3:
        return pc_bev.permute(1,2,0).to(device)
    else:
        ValueError('pc_bev dim is not 2 or 3')


# Generate BEV encoded by occupancy information with CUDA accelerate 
def generate_bev_occ(pc):
    '''
    pc: normalized point cloud
    '''     
    # returns cfg.num_height * cfg.num_ring * cfg.num_sector matrix
    size = pc.shape[0]
    pc = pc[:,:3]
    pc_bev = np.zeros([cfg.num_height * cfg.num_ring * cfg.num_sector])
    pc = pc.transpose().flatten().astype(np.float32)

    transer_bev = voxelocc.GPUTransformer(pc, size, cfg.max_length, cfg.max_height, cfg.num_ring, cfg.num_sector, cfg.num_height, 1)
    transer_bev.transform()
    point_t_bev = transer_bev.retreive()
    point_t_bev = point_t_bev.reshape(-1, 3)
    point_t_bev = point_t_bev[...,2]

    pc_bev = point_t_bev.reshape(cfg.num_height, cfg.num_ring, cfg.num_sector)
    pc_bev = torch.from_numpy(pc_bev).to(device)
    return pc_bev


# Generate BEV encoded by point features with CUDA accelerate 
def generate_bev_feats(pc):
    '''
    pc: normalized point cloud
    ''' 
    # returns cfg.num_height * num_feats * cfg.num_ring * cfg.num_sector matrix    
    size = pc.shape[0]
    pc = pc[:,:3]

    ###### KNN ######
    k = 30  # num of neighbors
    k_indices, k_covs, k_entropy, k_eigens_, k_vectors_  = build_neighbors_NN(pc, k)

    ###### LPD-Net Features ######
    k_indices = k_indices.flatten().astype(np.int32)
    k_eigens_ = k_eigens_.flatten()
    pc_flatten = pc.flatten()

    featureExtractor = voxelfeat.GPUFeatureExtractor(pc_flatten, size, 13, 30, k_indices, k_eigens_)
    GPUfeat = featureExtractor.get_features()
    GPUfeat = GPUfeat.reshape(-1,13)
    GPUfeat = GPUfeat.take([0,1,3,10,11,12], axis=1)  # [0,1,3,10,11,12] x 5,7,8,9
    pc = np.concatenate((pc, GPUfeat), axis=1)  # num_points * 13 
    featsize = pc.shape[1]

    pc = pc.transpose().flatten().astype(np.float32)

    transer_bev = voxelfeat.GPUTransformer(pc, size, cfg.max_length, cfg.max_height, cfg.num_ring, cfg.num_sector, cfg.num_height, featsize)
    transer_bev.transform()
    point_t_bev = transer_bev.retreive()
    point_t_bev = point_t_bev.reshape(-1, featsize)
    point_t_bev = point_t_bev[...,3:]
    pc_bev = point_t_bev.reshape(cfg.num_height, cfg.num_ring, cfg.num_sector, featsize-3)
    
    pc_bev = torch.from_numpy(pc_bev).to(device)
    pc_bev = pc_bev.squeeze(0)
    pc_bev = pc_bev.permute(2,0,1)
    return pc_bev
############ BEV Features Generation ############


############ RING Descriptor Generation ############
#  GPU version of RING descriptor generation
def generate_RING(pc_bev):
    angles = torch.FloatTensor(np.linspace(0, 2*np.pi, cfg.num_ring).astype(np.float32))
    radon = ParallelBeam(cfg.num_sector, angles)

    pc_RING = radon.forward(pc_bev)
    pc_TIRING, _ = forward_row_fft(pc_RING)

    return pc_RING, pc_TIRING


#  CPU version of RING descriptor generation
def generate_RING_cpu(pc_bev):
    angles = np.linspace(0., 360., cfg.num_ring, endpoint=False).astype(np.float32)
    pc_RING = torch.from_numpy(radon(pc_bev.numpy().squeeze(0), theta=angles)).to(device).unsqueeze(0)
    pc_RING = pc_RING.transpose(1, 2)

    pc_TIRING, _ = forward_row_fft(pc_RING)

    return pc_RING, pc_TIRING
############ RING Descriptor Generation ############



############ Pose Estimation ############
# Compute the max correlation value and the corresponding circular shift 
def fast_corr(a, b, zero_mean_normalize=True):
    if zero_mean_normalize:
        a = fn.normalize(a, mean=a.mean(), std=a.std())
        b = fn.normalize(b, mean=b.mean(), std=b.std())
    else:
        a = F.normalize(a, dim=(-2,-1))
        b = F.normalize(b, dim=(-2,-1))    
    a_fft = torch.fft.fft2(a, dim=-2, norm="ortho")
    b_fft = torch.fft.fft2(b, dim=-2, norm="ortho")
    corr = torch.fft.ifft2(a_fft*b_fft.conj(), dim=-2, norm="ortho")  
    corr = torch.sqrt(corr.real**2 + corr.imag**2)
    corr = torch.sum(corr, dim=-3) # add this line for multi feature channels
    corr = torch.sum(corr, dim=-1).view(-1, cfg.num_ring) 
    
    corr = torch.fft.fftshift(corr, dim=-1)
    angle = cfg.num_ring//2 - torch.argmax(corr, dim=-1)
    max_corr = torch.max(corr, dim=-1)[0]
    dist = 1 - max_corr/(0.15*a.shape[-3]*a.shape[-2]*a.shape[-1])
    dist = dist.cpu().numpy().squeeze()
    angle = angle.cpu().numpy().squeeze()

    return dist, angle


def batch_circorr(a, b, scale=1):
    circorr = circorr2(is_circular=True, zero_mean_normalize=True)
    corr, score, angle = circorr(a, b, scale)
    dist = 1 - score
    dist = dist.cpu().numpy()
    angle = angle.cpu().numpy()

    return dist, angle


# Calculate the circular shift to compensate the yaw difference between two RINGs
def calculate_row_shift(shift):
    if shift < cfg.num_ring // 2:
        shift = -shift 
    else:
        shift = shift - cfg.num_ring
    return shift


# Rotate tensor image with torch
def rotate_bev(image: torch.Tensor, angle: float):
    # convert angle to degree
    angle = angle * 180 / np.pi
    image_rotated = fn.rotate(image, angle)
    return image_rotated


# Rotate numpy image with cv2
def rotate_image(image: np.ndarray, angle: float):
    col, row = cfg.num_ring, cfg.num_sector
    image_center = (col // 2, row // 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    # result = cv2.warpAffine(image, rot_mat,  (col, row), flags=cv2.INTER_LINEAR)
    result = cv2.warpAffine(image, rot_mat, (col, row))
    # plt.imshow(result)
    # plt.show()
    return result


# Exhaustive search for the best translation
def solve_translation(a, b, zero_mean_normalize=True):
    if zero_mean_normalize:
        a = fn.normalize(a, mean=a.mean(), std=a.std())
        b = fn.normalize(b, mean=b.mean(), std=b.std())
    else:
        a = F.normalize(a, dim=(-2,-1))
        b = F.normalize(b, dim=(-2,-1))  
    # 2D cross correlation
    a_fft = torch.fft.fft2(a, dim=(-2,-1), norm="ortho")
    b_fft = torch.fft.fft2(b, dim=(-2,-1), norm="ortho")
    corr = torch.fft.ifft2(a_fft*b_fft.conj(), dim=(-2,-1), norm="ortho")  
    corr = torch.sqrt(corr.real**2 + corr.imag**2)
    corr = torch.sum(corr, dim=-3) # add this line for multi feature channels
    # print((corr==torch.max(corr)).nonzero())

    # shift the correlation to the center
    corr = torch.fft.fftshift(corr, dim=(-2,-1))

    # get the index of the max correlation
    idx_x = (corr==torch.max(corr)).nonzero()[0][0]
    idx_y = (corr==torch.max(corr)).nonzero()[0][1]
    # print(idx_x, idx_y)    

    # get the translation value
    x = cfg.num_ring//2 - idx_x
    y = cfg.num_sector//2 - idx_y

    return x.cpu().numpy(), y.cpu().numpy(), -torch.max(corr).cpu().numpy()
############ Pose Estimation ############