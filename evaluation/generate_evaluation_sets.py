import os
import pickle
import argparse
import numpy as np
from typing import List, Dict
from sklearn.neighbors import KDTree

from datasets.NCLTDataset import NCLTSequence, NCLTSequences
from datasets.KITTIDataset import KITTISequence, KITTISequences
from datasets.MulRanDataset import MulRanSequence, MulRanSequences
from datasets.OxfordRadarDataset import OxfordRadarSequence, OxfordRadarSequences


class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array, pose: np.array):
        # position: x, y position in meters
        # pose: 6 DoF pose (as 4x4 pose matrix)
        assert position.shape == (2,)
        assert pose is None or pose.shape == (4, 4)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position
        self.pose = pose

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position, self.pose


class EvaluationSet:
    # Evaluation set consisting of map and query elements
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))

        self.map_set = []
        for e in map_l:
            self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))

    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions

    def get_map_poses(self):
        # Get map positions as (N, 2) array
        poses = np.zeros((len(self.map_set), 4, 4), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            poses[ndx] = pos.pose
        return poses

    def get_query_poses(self):
        # Get query positions as (N, 2) array
        poses = np.zeros((len(self.query_set), 4, 4), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            poses[ndx] = pos.pose
        return poses
    

def get_scans(sequence) -> List[EvaluationTuple]:
    # Get a list of all readings from the test area in the sequence
    elems = []
    for ndx in range(len(sequence)):
        pose = sequence.poses[ndx]
        position = sequence.xys[ndx]
        item = EvaluationTuple(sequence.timestamps[ndx], sequence.filepaths[ndx], position=position, pose=pose)
        elems.append(item)
    return elems


def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple],
                          dist_threshold: float) -> List[EvaluationTuple]:
    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position

    # Build a kdtree
    kdtree = KDTree(map_pos)

    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] radius")
    return filtered_query_set


def generate_evaluation_set(dataset: str, dataset_root: str, map_sequence: str, query_sequence: str, split: str = 'test', 
map_sampling_distance: float = 20.0, query_sampling_distance: float = 5.0, dist_threshold: float = 20.0):
    assert split in ['test', 'all']
    if dataset == "nclt":
        map = NCLTSequence(dataset_root, sequence_name=map_sequence, split=split, sampling_distance=map_sampling_distance)
        query = NCLTSequence(dataset_root, sequence_name=query_sequence, split=split, sampling_distance=query_sampling_distance)
    elif dataset == "mulran":
        map = MulRanSequence(dataset_root, sequence_name=map_sequence, split=split, sampling_distance=map_sampling_distance)
        query = MulRanSequence(dataset_root, sequence_name=query_sequence, split=split, sampling_distance=query_sampling_distance)
    elif dataset == "kitti":
        map = KITTISequence(dataset_root, sequence_name=map_sequence, split=split, sampling_distance=map_sampling_distance)
        query = KITTISequence(dataset_root, sequence_name=query_sequence, split=split, sampling_distance=query_sampling_distance)
    elif dataset == "oxford_radar":
        map = OxfordRadarSequence(dataset_root, sequence_name=map_sequence, split=split, sampling_distance=map_sampling_distance)
        query = OxfordRadarSequence(dataset_root, sequence_name=query_sequence, split=split, sampling_distance=query_sampling_distance)
    else:
        raise ValueError('dataset is not "nclt" or "mulran" or "kitti" or "oxford_radar"')

    map_set = get_scans(map)
    query_set = get_scans(query)

    # Filters out query elements without a corresponding map element within dist_threshold threshold
    query_set = filter_query_elements(query_set, map_set, dist_threshold)

    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nclt", help="dataset type (nclt / mulran / kitti / oxford_radar)")
    parser.add_argument("--dataset_root", type=str, default="./data/NCLT", help="path to the dataset root")
    parser.add_argument("--map_sequence", type=str, default="2012-02-04", help="map sequence for loop closure detection")
    parser.add_argument("--query_sequence", type=str, default="2012-03-17", help="query sequence for loop closure detection")
    parser.add_argument("--map_sampling_distance", type=float, default=20.0, help="map sampling distance in meters")
    parser.add_argument("--query_sampling_distance", type=float, default=5.0, help="query sampling distance in meters")
    parser.add_argument("--dist_threshold", type=float, default=20.0, help="revisit threshold in meters")
    args = parser.parse_args()
    
    print(f'Dataset: {args.dataset}')
    print(f'Dataset root: {args.dataset_root}')
    print(f'Map sequence: {args.map_sequence}')
    print(f'Query sequence: {args.query_sequence}')
    print(f'Map sampling displacement between consecutive anchors: {args.map_sampling_distance}')
    print(f'Query sampling displacement between consecutive anchors: {args.query_sampling_distance}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

    test_set = generate_evaluation_set(args.dataset, args.dataset_root, map_sequence=args.map_sequence, query_sequence=args.query_sequence, 
        map_sampling_distance=args.map_sampling_distance, query_sampling_distance=args.query_sampling_distance, dist_threshold=args.dist_threshold)

    pickle_name = f'test_{args.map_sequence}_{args.query_sequence}_{args.map_sampling_distance}_{args.query_sampling_distance}_{args.dist_threshold}.pickle'
    file_path_name = os.path.join(args.dataset_root, pickle_name)
    test_set.save(file_path_name)
