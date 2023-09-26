'''
Parameters of RING for Loop Detection
'''

# Sampling Gap
train_sampling_gap = 10 # in meters
test_sampling_gap = 10 # in meters

# Top k Loop Detection 
top_k = 10

# Formation Loop Detection Window Size
window_size = 5

# Batch Size of Correlation Calculation
batch_size = 16

# Point Cloud Process
point_cloud = {
    'x_bound': [-70.0, 70.0],
    'y_bound': [-70.0, 70.0],
    'z_bound': [1.0, 20.0]
}

# BEV Resolution
num_ring = 120
num_sector = 120
num_height = 1
max_length = 1
max_height = 1

# Place Recognition Parameters
search_ratio = 0.1
num_candidates = 10
exclude_recent_nodes = 30
dist_threshold = 0.48 # descriptor distance (correlation) threshold for place recognition
revisit_criteria = 10 # in meters

# ICP Parameters
max_icp_iter = 100
icp_tolerance = 0.001
icp_max_distance = 5.0
num_icp_points = 6000
icp_fitness_score = 0.12