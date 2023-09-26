'''
	Pre-processing: prepare_data in LPD-Net
	generate KNN neighborhoods and calculate feature as the feature matrix of point
	Reference: LPD-Net: 3D Point Cloud Learning for Large-Scale Place Recognition and Environment Analysis, ICCV 2019
	author: Chuanzhe Suo(suo_ivy@foxmail.com)
	created: 10/26/18
'''
                        
import os
import sys
import time 
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KDTree

np.seterr(divide='ignore',invalid='ignore')

def calculate_entropy_array(eigen):
    L_ = (eigen[:,0] - eigen[:,1]) / eigen[:,0]
    P_ = (eigen[:,1] - eigen[:,2]) / eigen[:,0]
    S_ = eigen[:,2] / eigen[:,0]
    Entropy = -L_*np.log(L_)-P_*np.log(P_)-S_*np.log(S_)
    return Entropy


def covariation_eigenvalue(pointcloud, neighborhood_index):
    ### calculate covariation and eigenvalue of 3D and 2D
    # prepare neighborhood
    neighborhoods = pointcloud[neighborhood_index]

    # 3D cov and eigen by matrix
    Ex = np.average(neighborhoods, axis=1)
    Ex = np.reshape(np.tile(Ex,[neighborhoods.shape[1]]), neighborhoods.shape)
    P = neighborhoods-Ex
    cov_ = np.matmul(P.transpose((0,2,1)),P)/(neighborhoods.shape[1]-1)

    eigen_ = torch.linalg.eigvalsh(torch.from_numpy(cov_))
    eigen_ = eigen_.cpu().numpy()
    indices = np.argsort(eigen_)
    indices = indices[:,::-1]
    pcs_num_ = eigen_.shape[0]
    indx = np.reshape(np.arange(pcs_num_), [-1, 1])
    eig_ind = indices + indx*3
    vec_ind = np.reshape(eig_ind*3, [-1,1]) + np.full((pcs_num_*3,3), [0,1,2])
    vec_ind = np.reshape(vec_ind, [-1,3,3])
    eigen3d_ = np.take(eigen_, eig_ind)
    vectors_ = []
    entropy_ = calculate_entropy_array(eigen3d_)

    # 2d cov and eigen
    cov2d_ = cov_[:,:2,:2]
    eigen2d = torch.linalg.eigvalsh(torch.from_numpy(cov2d_))
    eigen2d = eigen2d.cpu().numpy()
    indices = np.argsort(eigen2d)
    indices = indices[:, ::-1]
    pcs_num_ = eigen2d.shape[0]
    indx = np.reshape(np.arange(pcs_num_), [-1, 1])
    eig_ind = indices + indx * 2
    eigen2d_ = np.take(eigen2d, eig_ind)

    eigens_ = np.append(eigen3d_,eigen2d_,axis=1)

    return cov_, entropy_, eigens_, vectors_


def build_neighbors_NN(pointcloud, k):
	### using KNN NearestNeighbors cluster according k
    start = time.time()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1).fit(pointcloud)
    distances, indices = nbrs.kneighbors(pointcloud)
    end = time.time()
    # print("neighbors time: ", end-start)
    
    start2 = time.time()
    covs, entropy, eigens_, vectors_ = covariation_eigenvalue(pointcloud, indices)
    end2 = time.time()
    # print("covariation_eigenvalue time: ", end2-start2)

    return indices, covs, entropy, eigens_, vectors_


def calculate_features(pointcloud, nbrs_index, eigens_, vectors_):
    ### calculate handcraft feature with eigens and statistics data

    # features using eigens
    eig3d = eigens_[:3]
    eig2d = eigens_[3:5]

    # 3d
    C_ = eig3d[2] / (eig3d.sum()) # change of curvature
    O_ = np.power((eig3d.prod() / np.power(eig3d.sum(), 3)), 1.0 / 3) # omni-variance
    # L_ = (eig3d[0] - eig3d[1]) / eig3d[0]
    E_ = -((eig3d / eig3d.sum()) * np.log(eig3d / eig3d.sum())).sum() # eigenvalue-entropy
    # P_ = (eig3d[1] - eig3d[2]) / eig3d[0]
    # S_ = eig3d[2] / eig3d[0]
    # A_ = (eig3d[0] - eig3d[2]) / eig3d[0]
    # X_ = eig3d.sum()
    # D_ = 3 * nbrs_index.shape[0] / (4 * math.pi * eig3d.prod())
    # 2d
    # S_2 = eig2d.sum() # 2D scattering
    L_2 = eig2d[1] / eig2d[0] # 2D linearity
    # features using statistics data
    neighborhood = pointcloud[nbrs_index]
    nbr_dz = neighborhood[:, 2] - neighborhood[:, 2].min()
    dZ_ = nbr_dz.max() # Fz features - maximum height difference
    vZ_ = np.var(nbr_dz) # Fz features - height variance
    # V_ = vectors_[2][2]
    # not ok: L_, D_, S_2, V_
    features = np.asarray([C_, O_, E_, L_2, dZ_, vZ_]) #([C_,O_,L_,E_,D_,S_2,L_2,dZ_,vZ_,V_])
    # features = np.asarray([dZ_])
    return features


def get_pointfeat(pointcloud):
    k = 30
    k_indices, k_covs, k_entropy, k_eigens_, k_vectors_  = build_neighbors_NN(pointcloud, k)
    start2 = time.time()

    points_feature = []
    for index in range(pointcloud.shape[0]):
        neighborhood = k_indices[index]
        eigens_ = k_eigens_[index]
        vectors_ = k_vectors_[index]

        # calculate point feature
        feature = calculate_features(pointcloud, neighborhood, eigens_, vectors_)
        points_feature.append(feature)

    end2 = time.time()
    print('calculate_features time:', end2-start2)

    points_feature = np.asarray(points_feature)

    return points_feature