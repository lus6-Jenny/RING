# utils for point cloud processing

import copy
import torch
import random
import pygicp
import numpy as np
import open3d as o3d
import utils.config as cfg


# Preprocess and normalize the point cloud 
def load_pc_infer(pc):
    # returns Nx3 matrix
    pc = np.array(pc, dtype=np.float32)
    idx = (pc[:,0] > cfg.point_cloud["x_bound"][0]) * (pc[:,0] < cfg.point_cloud["x_bound"][1]) * (pc[:,1] > cfg.point_cloud["y_bound"][0]) * \
          (pc[:,1] < cfg.point_cloud["y_bound"][1]) * (pc[:,2] > cfg.point_cloud["z_bound"][0]) * (pc[:,2] < cfg.point_cloud["z_bound"][1])

    pc = pc[idx]
    pc_filtered = copy.deepcopy(pc)
    pc[:,0] = pc[:,0] / cfg.point_cloud["x_bound"][1]
    pc[:,1] = pc[:,1] / cfg.point_cloud["y_bound"][1]
    pc[:,2] = (pc[:,2] - cfg.point_cloud["z_bound"][0]) / (cfg.point_cloud["z_bound"][1] - cfg.point_cloud["z_bound"][0])
    return pc_filtered, pc   


# Randomly downsample the pointcloud to a certain number of points
def random_sampling(orig_points, num_points):
    assert orig_points.shape[0] > num_points
    points_down_idx = random.sample(range(orig_points.shape[0]), num_points)
    down_points = orig_points[points_down_idx, :]

    return down_points


# Apply random rotation along z axis
def random_rotation(xyz, angle_range=(-np.pi, np.pi)):
    angle = np.random.uniform(*angle_range)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]]).transpose()
    return np.dot(xyz, rotation_matrix)


# Apply 4x4 SE(3) transformation matrix on (N, 3) point cloud or 3x3 transformation on (N, 2) point cloud
def apply_transform(pc: torch.Tensor, m: torch.Tensor):
    assert pc.ndim == 2
    n_dim = pc.shape[1]
    assert n_dim == 2 or n_dim == 3
    assert m.shape == (n_dim + 1, n_dim + 1)
    # (m @ pc.t).t = pc @ m.t
    pc = pc @ m[:n_dim, :n_dim].transpose(1, 0) + m[:n_dim, -1]

    return pc


# Make up feature with open3d
def make_open3d_feature(data, dim, npts):
    feature = o3d.pipelines.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.cpu().numpy().astype('d').transpose()
    return feature


# Make up point cloud with open3d
def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def draw_pc(pc):
    '''
    pc: np.ndarray
    '''
    pc = copy.deepcopy(pc)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc)    
    pcd1.paint_uniform_color([1, 0.706, 0])
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
    o3d.visualization.draw_geometries([pcd1, coord],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def draw_pc_pair(pc1, pc2):
    '''
    pc1 & pc2: np.ndarray
    '''    
    pc1 = copy.deepcopy(pc1)
    pc2 = copy.deepcopy(pc2)    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)    
    pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
    o3d.visualization.draw_geometries([pcd1, pcd2, coord],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])   


def draw_registration_result(source, target, transformation):
    '''
    source & target: np.ndarray
    '''        
    source = copy.deepcopy(source)
    target = copy.deepcopy(target)    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(source)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(target)    
    pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])
    pcd1.transform(transformation)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])    
    o3d.visualization.draw_geometries([pcd1, pcd2, coord],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])                              


# fast_gicp (https://github.com/SMRT-AIST/fast_gicp)
def fast_gicp(source, target, max_correspondence_distance=1.0, init_pose=np.eye(4)):
    # downsample the point cloud before registration
    source = pygicp.downsample(source, 0.25)
    target = pygicp.downsample(target, 0.25)

    # pygicp.FastGICP has more or less the same interfaces as the C++ version
    gicp = pygicp.FastGICP()
    gicp.set_input_target(target)
    gicp.set_input_source(source)

    # optional arguments
    gicp.set_num_threads(4)
    gicp.set_max_correspondence_distance(max_correspondence_distance)

    # align the point cloud using the initial pose calculated by RING
    T_matrix = gicp.align(initial_guess=init_pose)

    # get the fitness score
    fitness = gicp.get_fitness_score(max_range=1.0)
    # get the transformation matrix
    T_matrix = gicp.get_final_transformation()

    return fitness, T_matrix


# open3d icp
def o3d_icp(source, target, transform: np.ndarray = None, point2plane: bool = False,
        inlier_dist_threshold: float = 1.0, max_iteration: int = 200):
    # transform: initial alignment transform
    if transform is not None:
        transform = transform.astype(float)

    voxel_size = 0.25
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(source)
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(target)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    if point2plane:
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    if transform is not None:
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold, transform,
                                                              estimation_method=transform_estimation,
                                                              criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    else:
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold,
                                                              estimation_method=transform_estimation,
                                                              criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return reg_p2p.fitness, reg_p2p.transformation, reg_p2p.inlier_rmse


class RandomOcclude:
    def __init__(self, ang):
        assert 0 <= ang <= 360, f'occluded angle must be in [0, 360] range, is {ang}'
        self.ang = ang

    def __call__(self, pc):
        # centroid = np.mean(pc, axis=0)
        # pc -= centroid
        ang = np.deg2rad(self.ang)
        start_ang = np.random.rand(1) * 2 * np.pi
        start_ang = np.where(start_ang < 0, start_ang + 2 * np.pi, start_ang)
        end_ang = start_ang + ang
        end_ang = np.where(end_ang < 0, end_ang + 2 * np.pi, end_ang)
        angles = np.arctan2(pc[:, 1], pc[:, 0])
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)
        mask = np.logical_or(angles < start_ang, angles > end_ang)
        pc = pc[mask, :]
        # pc += centroid
        return pc