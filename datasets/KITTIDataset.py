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
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset, ConcatDataset, DataLoader


# calibration matrix
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    cam_to_velo = np.reshape(data['Tr'], (3, 4))
    cam_to_velo = np.vstack([cam_to_velo, [0, 0, 0, 1]])
    return cam_to_velo


def load_pc_kitti(file_pathname: str):
    # Load point cloud from file
    pc = np.fromfile(file_pathname, dtype=np.float32)
    pc = np.reshape(pc, (-1, 4)) # x,y,z,intensity
    pc = pc[:,:3]

    return pc


class KITTIPointCloudLoader:
    def __init__(self, remove_zero_points: bool = True, remove_ground_plane: bool = True, ground_plane_level: float = -1.0):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = remove_zero_points
        self.remove_ground_plane = remove_ground_plane
        self.ground_plane_level = ground_plane_level

    def read_pc(self, file_pathname: str):
        # Reads the point cloud and preprocess (optional removal of zero points and points on the ground plane
        pc = load_pc_kitti(file_pathname)
        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]        
        
        pc = np.unique(pc, axis=0) # remove duplicate points  

        return pc
    
    def normalize_pc(self, pc, xbound: float = cfg.point_cloud["x_bound"][1], ybound: float = cfg.point_cloud["y_bound"][1], zbound: float = 10.):
        # Normalize the point cloud
        pc = np.array(pc, dtype=np.float32)
        pc = pc[:,:3]

        mask = (np.abs(pc[:,0]) < xbound) * (np.abs(pc[:,1]) < ybound) * (pc[:,2] > self.ground_plane_level) * (pc[:,2] < zbound)

        pc = pc[mask]
        pc[:,0] = pc[:,0] / xbound
        pc[:,1] = pc[:,1] / ybound
        pc[:,2] = pc[:,2] / zbound 

        return pc   
    

class KITTISequence(Dataset):
    """
    Single KITTI sequence indexed as a single dataset containing point clouds and poses
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
        self.pose_file = os.path.join(self.sequence_path, 'poses.txt')
        assert os.path.exists(self.pose_file), f'Cannot access global pose file: {self.pose_file}'
        self.lidar_path = os.path.join(self.sequence_path, 'velodyne')
        assert os.path.exists(self.lidar_path), f'Cannot access lidar scans: {self.lidar_path}'
        # self.lidar_files = sorted(glob.glob(os.path.join(self.lidar_path, '*.bin')))
        self.pc_loader = KITTIPointCloudLoader()
        self.times_file = os.path.join(self.sequence_path, 'times.txt')    
        assert os.path.exists(self.times_file), f'Cannot access sequence times file: {self.times_file}'
        self.calib_file = os.path.join(self.sequence_path, 'calib.txt')     
        assert os.path.exists(self.calib_file), f'Cannot access sequence calib file: {self.calib_file}'
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
        # Read global poses from .txt file and link each lidar_scan with the nearest pose
        # threshold: timestamp threshold in seconds
        # Returns a dictionary with (4, 4) pose matrix indexed by a timestamp (as integer)
        calib = read_calib_file(self.calib_file)   
        with open(self.pose_file, "r") as h:
            pose_data = h.readlines()

        n = len(pose_data)
        print(f'Number of global poses: {n}')
        system_timestamps = np.genfromtxt(self.times_file)
        poses = np.zeros((n, 4, 4), dtype=np.float32)       # 4x4 pose matrix

        for ndx, pose in enumerate(pose_data):
            # Split by comma and remove whitespaces
            temp = [e.strip() for e in pose.split(' ')]
            assert len(temp) == 12, f'Invalid line in global poses file: {temp}'
            # poses in kitti are in cam0 reference
            poses[ndx] = np.array([[float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])],
                                    [float(temp[4]), float(temp[5]), float(temp[6]), float(temp[7])],
                                    [float(temp[8]), float(temp[9]), float(temp[10]), float(temp[11])],
                                    [0., 0., 0., 1.]])
            poses[ndx] = np.linalg.inv(calib) @ (poses[ndx] @ calib)

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
        
        # Split data into train / test set
        if self.split != 'all':      
            if self.split == 'train':
                mask = check_in_train_set(lidar_xys, dataset='kitti')
            elif self.split == 'test':
                mask = check_in_test_set(lidar_xys, dataset='kitti')
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


class KITTISequences(Dataset):
    """
    Multiple NCLT sequences indexed as a single dataset containing point clouds and poses
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
            sequence = KITTISequence(dataset_root, sequence_name, split, sampling_distance)
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
    base_path = './data/KITTI/'
    folder = '08' # sequence folder for training   
    train_dataset = KITTISequence(base_path, folder, split='train')
    test_dataset = KITTISequence(base_path, folder, split='test')
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
    