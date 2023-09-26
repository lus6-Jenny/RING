import os
import sys
import time
import torch
import numpy as np
from utils.core import *
from utils.icp import icp
import utils.config as cfg
from utils.tools import imshow
from utils.poses import xyz_ypr2m, m2xyz_ypr
from utils.point_clouds import load_pc_infer, fast_gicp, draw_pc, draw_pc_pair, draw_registration_result

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print('BASE_DIR', BASE_DIR)
sys.path.append(BASE_DIR)


# Apply 4x4 transformation matrix to point cloud
def apply_transform(pc, transform): 
    pc_tranformed = np.dot(pc, transform[:3, :3].T) + transform[:3, 3]
    pc_tranformed = pc_tranformed.astype(np.float32)

    return pc_tranformed


if __name__ == '__main__':
    pc1 = np.load('./test.npy')
    gt_x = np.random.uniform(-10, 10)
    gt_y = np.random.uniform(-10, 10)
    gt_yaw = np.random.uniform(0, 2*np.pi)
    transform = xyz_ypr2m(gt_x, gt_y, 0, gt_yaw, 0, 0)
    pc2 = apply_transform(pc1, transform)

    pc1, pc1_norm = load_pc_infer(pc1)
    pc2, pc2_norm = load_pc_infer(pc2)

    # draw_pc(pc1)
    # draw_pc(pc2)
    # draw_pc_pair(pc1, pc2)
    
    # ------------ RING Generation ------------
    bev_type = "occ" # "occ" or "feat"
    # generate BEV features
    times = time.time()
    if device == torch.device("cpu"):
        pc1_bev = generate_bev(pc1) # BEV encoded by occupancy information
        pc2_bev = generate_bev(pc2) # BEV encoded by occupancy information
    else:      
        if bev_type == "occ":
            pc1_bev = generate_bev_occ(pc1_norm) # BEV encoded by occupancy information
            pc2_bev = generate_bev_occ(pc2_norm) # BEV encoded by occupancy information
        elif bev_type == "feat":
            pc1_bev = generate_bev_feats(pc1_norm) # BEV encoded by point features
            pc2_bev = generate_bev_feats(pc2_norm) # BEV encoded by point features
        else:
            raise ValueError('bev_type is not "occ" or "feat"')
    timee = time.time()
    # print("Time of BEV generation:", timee - times, 's')
    print("BEV shapes:", pc1_bev.shape, pc2_bev.shape)

    # generate RING and TIRING descriptors
    times = time.time()
    if device == torch.device("cuda:0"):
        pc1_RING, pc1_TIRING = generate_RING(pc1_bev)
        pc2_RING, pc2_TIRING = generate_RING(pc2_bev)        
    elif device == torch.device("cpu"):
        pc1_RING, pc1_TIRING = generate_RING_cpu(pc1_bev)
        pc2_RING, pc2_TIRING = generate_RING_cpu(pc2_bev)
    timee = time.time()
    # print("Time of RING generation:", timee - times, 's')
    # ------------ RING Generation ------------

    # ------------ Pose Estimation ------------
    ang_res = 2 * np.pi / cfg.num_ring # angular resolution
    dist, angle_matched = fast_corr(pc1_TIRING, pc2_TIRING)

    # angle between the two matched RINGs in grids
    angle_matched_extra = angle_matched - cfg.num_ring // 2
    # convert the matched angle from grids to radians
    angle_matched_rad = angle_matched * ang_res 
    angle_matched_extra_rad = angle_matched_extra * ang_res        

    bev_rotated = rotate_bev(pc1_bev, angle_matched_rad)
    bev_rotated_extra = rotate_bev(pc1_bev, angle_matched_extra_rad)
    
    # imshow(pc1_bev[0], 'bev1.png')
    # imshow(pc1_RING[0], 'sinogram1.png')
    # imshow(pc1_TIRING[0], 'ting1.png')
    # imshow(pc2_bev[0], 'bev2.png')
    # imshow(pc2_RING[0], 'sinogram2.png')
    # imshow(pc2_TIRING[0], 'ting2.png')
    # imshow(bev_rotated[0], 'bev_rotated.png')
    # imshow(bev_rotated_extra[0], 'bev_rotated_extra.png')

    # solve the translation between the two matched bevs
    x, y, error = solve_translation(bev_rotated, pc2_bev)
    x_extra, y_extra, error_extra = solve_translation(bev_rotated_extra, pc2_bev)

    if error < error_extra:
        trans_x = x / cfg.num_sector * (cfg.point_cloud["x_bound"][1] - cfg.point_cloud["x_bound"][0])  # in meters
        trans_y = y / cfg.num_ring * (cfg.point_cloud["y_bound"][1] - cfg.point_cloud["y_bound"][0])   # in meters
        rot_yaw = angle_matched_rad  # in radians
    else:
        trans_x = x_extra / cfg.num_sector * (cfg.point_cloud["x_bound"][1] - cfg.point_cloud["x_bound"][0])  # in meters
        trans_y = y_extra / cfg.num_ring * (cfg.point_cloud["y_bound"][1] - cfg.point_cloud["y_bound"][0])  # in meters 
        rot_yaw = angle_matched_extra_rad  # in radians
    
    init_pose = xyz_ypr2m(trans_x, trans_y, 0, rot_yaw, 0, 0)
    print("Ground truth translation: x: {}, y: {}, rotation: {}".format(gt_x, gt_y, gt_yaw))
    print("Estimated translation: x: {}, y: {}, rotation: {}".format(trans_x, trans_y, rot_yaw))
    draw_registration_result(pc1, pc2, init_pose)
    # ------------ Pose Estimation ------------

    # ------------ Pose Refinement ------------
    times = time.time()
    icp_fitness_score, loop_transform = fast_gicp(pc1, pc2, max_correspondence_distance=cfg.icp_max_distance, init_pose=init_pose)
    timee = time.time() 

    # print("ICP fitness score:", icp_fitness_score)       
    # print("ICP processed time:", timee - times, 's')
    x, y, z, yaw, pitch, roll = m2xyz_ypr(loop_transform)
    print("Refined translation: x: {}, y: {}, rotation: {}".format(x, y, yaw))
    draw_registration_result(pc1, pc2, loop_transform)
    # ------------ Pose Refinement ------------