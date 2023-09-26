import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import utils.config as cfg
from typing import List, Tuple
from utils.poses import xyz_ypr2m
from utils.tools import find_nearest_ndx, check_in_train_set, check_in_test_set
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, ConcatDataset, DataLoader


# calibrate the lidar data of mulran dataset
def get_mulran_calib():
    calib = np.eye(4)
    calib[0,3] = 1.7042
    calib[1,3] = -0.021
    calib[2,3] = 1.8047
    calib[:3,:3] = Rotation.from_euler('xyz',[0.0001,0.0003,179.6654],degrees=True).as_matrix()
    return calib


def load_pc_mulran(file_pathname: str):
    # Load point cloud from file
    n_vec = 4
    pc = np.fromfile(file_pathname, dtype=np.float32)
    pc = np.reshape(pc, (-1, n_vec))
    pc = pc[:,:3]
    return pc


class MulRanPointCloudLoader:
    def __init__(self, remove_zero_points: bool = True, remove_ground_plane: bool = True, ground_plane_level: float = -0.9):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = remove_zero_points
        self.remove_ground_plane = remove_ground_plane
        self.ground_plane_level = ground_plane_level

    def read_pc(self, file_pathname: str):
        # Reads the point cloud and preprocess (optional removal of zero points and points on the ground plane
        pc = load_pc_mulran(file_pathname)

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]        

        pc = np.unique(pc, axis=0) # remove duplicate points  

        return pc
    
    def normalize_pc(self, pc, xbound: float = cfg.point_cloud["x_bound"][1], ybound: float = cfg.point_cloud["y_bound"][1], zbound: float = 19.):
        # Normalize the point cloud
        pc = np.array(pc, dtype=np.float32)
        pc = pc[:,:3]

        mask = (np.abs(pc[:,0]) < xbound) * (np.abs(pc[:,1]) < ybound) * (pc[:,2] > self.ground_plane_level) * (pc[:,2] < zbound)

        pc = pc[mask]
        pc[:,0] = pc[:,0] / xbound
        pc[:,1] = pc[:,1] / ybound
        pc[:,2] = (pc[:,2] - self.ground_plane_level) / (zbound - self.ground_plane_level) 

        return pc   
    

class MulRanSequence(Dataset):
    """
    Single MulRan sequence indexed as a single dataset containing point clouds and poses
    """
    def __init__(self, dataset_root: str, sequence_name: str, split: str = 'train', sampling_distance: float = 0.2):
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test', 'all']
        self.dataset_root = dataset_root
        self.sequence_name = sequence_name
        self.sequence_path = os.path.join(self.dataset_root, self.sequence_name)
        assert os.path.exists(self.sequence_path), f'Cannot access sequence: {self.sequence_path}'
        self.split = split
        self.sampling_distance = sampling_distance
        # Maximum discrepancy between timestamps of LiDAR scan and global pose in seconds
        self.pose_time_tolerance = 1.
        self.pose_file = os.path.join(self.sequence_path, 'global_pose.csv')
        assert os.path.exists(self.pose_file), f'Cannot access global pose file: {self.pose_file}'
        self.lidar_path = os.path.join(self.sequence_path, 'Ouster')
        assert os.path.exists(self.lidar_path), f'Cannot access lidar scans: {self.lidar_path}'
        # self.lidar_files = sorted(glob.glob(os.path.join(self.lidar_path, '*.bin')))
        self.pc_loader = MulRanPointCloudLoader()
        self.timestamps, self.filepaths, self.poses, self.xys = self.get_scan_poses()
        assert len(self.filepaths) == len(self.poses), f'Number of lidar file paths and poses do not match: {len(self.filepaths)} vs {len(self.poses)}'
        print(f'{len(self.filepaths)} scans in {sequence_name}-{split}')
        # Build a kdtree based on X, Y position
        self.kdtree = KDTree(self.xys)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, ndx):
        pc_filepath = self.filepaths[ndx]
        pc = self.load_pc(pc_filepath)
        return {'ts': self.timestamps[ndx], 'pc': pc, 'position': self.xys[ndx], 'pose': self.poses[ndx]}

    def load_pc(self, scan_path):
        # Load point cloud from file
        pc = self.pc_loader.read_pc(scan_path)
        return pc
    
    def load_pcs(self, scan_paths):
        # Load point clouds from files
        pcs = []
        for scan_path in scan_paths:
            pc = self.load_pc(scan_path)
            if len(pc) == 0:
                continue
            pcs.append(pc)
        pcs = np.array(pcs)
        return pcs  
    
    def normalize_pc(self, pc):
        # Normalize the point cloud
        pc = self.pc_loader.normalize_pc(pc)
        return pc
    
    def normalize_pcs(self, pcs):
        # Normalize point clouds
        pcs = [self.normalize_pc(pc) for pc in pcs]
        return pcs
        
    def get_scan_poses(self):
        # Read global poses from .csv file and link each lidar_scan with the nearest pose
        # threshold: timestamp threshold in seconds
        # Returns a dictionary with (4, 4) pose matrix indexed by a timestamp (as integer)
        pose_data = pd.read_csv(self.pose_file, header=0, \
            names=['timestamp','x1','x2','x3','northing','y1','y2','y3','easting','z1','z2','z3','height'], low_memory=False)

        n = len(pose_data)
        print(f'Number of global poses: {n}')
        system_timestamps = np.zeros((n,), dtype=np.int64)
        poses = np.zeros((n, 4, 4), dtype=np.float32)       # 4x4 pose matrix
        calib = get_mulran_calib()
        calib = np.linalg.inv(calib)

        for ndx in range(n):
            row = pose_data.iloc[ndx]
            assert len(row) == 13, f'Invalid line in global poses file: {row}'
            ts = int(row['timestamp'])
            # transform to se3 matrix
            se3 = np.eye(4, dtype=np.float32)            
            se3[0,0] = float(row['x1'])
            se3[0,1] = float(row['x2'])
            se3[0,2] = float(row['x3'])
            se3[0,3] = float(row['northing'])
            se3[1,0] = float(row['y1'])
            se3[1,1] = float(row['y2'])
            se3[1,2] = float(row['y3'])
            se3[1,3] = float(row['easting'])
            se3[2,0] = float(row['z1'])
            se3[2,1] = float(row['z2'])
            se3[2,2] = float(row['z3'])
            se3[2,3] = float(row['height'])
            # add to system_timestamps and poses
            system_timestamps[ndx] = int(ts)
            poses[ndx] = se3 @ calib

        # Ensure timestamps and poses are sorted in ascending order
        sorted_ndx = np.argsort(system_timestamps, axis=0)
        system_timestamps = system_timestamps[sorted_ndx]
        poses = poses[sorted_ndx]

        # List LiDAR scan timestamps
        all_lidar_timestamps = [int(os.path.splitext(f)[0]) for f in os.listdir(self.lidar_path) if
                                os.path.splitext(f)[1] == '.bin']
        all_lidar_timestamps.sort()
        print(f'Number of global scans: {len(all_lidar_timestamps)}')

        lidar_timestamps = []
        lidar_filepaths = []
        lidar_poses = []
        count_rejected = 0

        for ndx, lidar_ts in enumerate(all_lidar_timestamps):
            # Find index of the closest timestamp
            closest_ts_ndx = find_nearest_ndx(lidar_ts, system_timestamps)
            delta = abs(system_timestamps[closest_ts_ndx] - lidar_ts)
            # Timestamp is in nanoseconds = 1e-9 second
            if delta > self.pose_time_tolerance * 1000000000:
                # Reject point cloud without corresponding pose
                count_rejected += 1
                continue

            lidar_timestamps.append(lidar_ts)
            lidar_filepaths.append(os.path.join(self.lidar_path, f'{lidar_ts}.bin'))
            lidar_poses.append(poses[closest_ts_ndx])
        
        lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)
        lidar_filepaths = np.array(lidar_filepaths)
        lidar_poses = np.array(lidar_poses, dtype=np.float32)     # 4x4 pose matrix
        lidar_xys = lidar_poses[:, :2, 3]                         # 2D position

        # split data into train / test set
        if self.split != 'all':      
            if self.split == 'train':
                mask = check_in_train_set(lidar_xys, dataset='mulran')
            elif self.split == 'test':
                mask = check_in_test_set(lidar_xys, dataset='mulran')
            lidar_timestamps = lidar_timestamps[mask]
            lidar_filepaths = lidar_filepaths[mask]
            lidar_poses = lidar_poses[mask]
            lidar_xys = lidar_xys[mask]

        # Sample lidar scans             
        prev_position = None
        mask = []
        for ndx, position in enumerate(lidar_xys):
            if prev_position is None:
                mask.append(ndx)
                prev_position = position
            else:
                displacement = np.linalg.norm(prev_position - position)
                if displacement > self.sampling_distance:
                    mask.append(ndx)
                    prev_position = position

        lidar_timestamps = lidar_timestamps[mask]
        lidar_filepaths = lidar_filepaths[mask]
        lidar_poses = lidar_poses[mask]
        lidar_xys = lidar_xys[mask]

        print(f'{len(lidar_timestamps)} scans with valid pose, {count_rejected} rejected due to unknown pose')
        return lidar_timestamps, lidar_filepaths, lidar_poses, lidar_xys

    def find_neighbours_ndx(self, position, radius):
        # Returns indices of neighbourhood point clouds for a given position
        assert position.ndim == 1
        assert position.shape[0] == 2
        # Reshape into (1, 2) axis
        position = position.reshape(1, -1)
        neighbours = self.kdtree.query_radius(position, radius)[0]
        return neighbours.astype(np.int32)


class MulRanSequences(Dataset):
    """
    Multiple MulRan sequences indexed as a single dataset containing point clouds and poses
    """
    def __init__(self, dataset_root: str, sequence_names: List[str], split: str = 'train', sampling_distance: float = 0.2):
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test', 'all']
        self.dataset_root = dataset_root
        self.sequence_names = sequence_names
        self.split = split
        self.sampling_distance = sampling_distance
        
        # Load all sequences
        sequences = []
        for sequence_name in sequence_names:
            sequence = MulRanSequence(dataset_root, sequence_name, split, sampling_distance)
            sequences.append(sequence)
        self.dataset = ConcatDataset(sequences)
        self.pc_loader = sequences[0].pc_loader

        # Concatenate all sequences
        self.timestamps = np.concatenate([s.timestamps for s in sequences])
        self.filepaths = np.concatenate([s.filepaths for s in sequences])
        self.poses = np.concatenate([s.poses for s in sequences])
        self.xys = np.concatenate([s.xys for s in sequences])
        assert len(self.filepaths) == len(self.poses), f'Number of lidar file paths and poses do not match: {len(self.filepaths)} vs {len(self.poses)}'
        print(f'{len(self.filepaths)} scans in {sequence_names}-{split}')

        # Build a kdtree based on X, Y position
        self.kdtree = KDTree(self.xys)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ndx):
        return self.dataset[ndx]

    def load_pc(self, scan_path):
        # Load point cloud from file
        pc = self.pc_loader.read_pc(scan_path)
        return pc
    
    def load_pcs(self, scan_paths):
        # Load point clouds from files
        pcs = []
        for scan_path in scan_paths:
            pc = self.load_pc(scan_path)
            if len(pc) == 0:
                continue
            pcs.append(pc)
        pcs = np.array(pcs)
        return pcs  
    
    def normalize_pc(self, pc):
        # Normalize the point cloud
        pc = self.pc_loader.normalize_pc(pc)
        return pc
    
    def normalize_pcs(self, pcs):
        # Normalize point clouds
        pcs = [self.normalize_pc(pc) for pc in pcs]
        return pcs

    def find_neighbours_ndx(self, position, radius):
        # Returns indices of neighbourhood point clouds for a given position
        assert position.ndim == 1
        assert position.shape[0] == 2
        # Reshape into (1, 2) axis
        position = position.reshape(1, -1)
        neighbours = self.kdtree.query_radius(position, radius)[0]
        return neighbours.astype(np.int32)


if __name__ == "__main__":
    # load dataset
    base_path = './data/MulRan/'
    folder = 'Sejong01' # sequence folder for training    
    train_dataset = MulRanSequence(base_path, folder, split='train')
    test_dataset = MulRanSequence(base_path, folder, split='test')
    train_positions = train_dataset.xys
    test_positions = test_dataset.xys
    # plot the splited results
    plt.scatter(train_positions[:,0], train_positions[:,1], c='r', s=0.1)
    plt.scatter(test_positions[:,0], test_positions[:,1], c='b', s=0.1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['train', 'test'])
    plt.title(f'{folder} Trajectory Split')
    plt.savefig(f'{folder}_split.png')
    plt.show()
       