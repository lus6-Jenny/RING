import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'font.weight': "bold"})
matplotlib.rcParams.update({'axes.labelweight': "bold"})
matplotlib.rcParams.update({'axes.titleweight': "bold"})
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.neighbors import KDTree


def calculate_dist(pose1, pose2):
    dist = np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)
    return dist


def compute_PR_pairs(pair_dists, query_positions, map_positions, thresholds, save_path, revisit_threshold: float = 10.0):

    tree = KDTree(map_positions)
    num_thresholds = len(thresholds)
    num_hits = np.zeros(num_thresholds) 
    num_false_alarms = np.zeros(num_thresholds) 
    num_correct_rejections = np.zeros(num_thresholds) 
    num_misses = np.zeros(num_thresholds)     
    
    for i in range(pair_dists.shape[0]):
        dist = pair_dists[i]
        query_position = query_positions[i]
        indices = tree.query_radius(query_position.reshape(1,-1), revisit_threshold)[0]
        for j, thre in enumerate(thresholds):
            if(min(dist) < thre):
                # if under the theshold, it is considered seen.
                # and then check the correctness
                real_dist = calculate_dist(query_position, map_positions[np.argmin(dist)])
                if real_dist < revisit_threshold:
                    # TP: Hit
                    num_hits[j] = num_hits[j] + 1
                else:
                    # FP: False Alarm 
                    num_false_alarms[j] = num_false_alarms[j] + 1                         
            
            else:
                if len(indices) == 0:
                    # TN: Correct rejection
                    num_correct_rejections[j] = num_correct_rejections[j] + 1
                else:           
                    # FN: MISS
                    num_misses[j] = num_misses[j] + 1 
    
    precisions = num_hits / (num_hits + num_false_alarms + 1e-10)
    recalls = num_hits / (num_hits + num_misses + 1e-10)
    precisions[np.isnan(precisions)] = 1.0
    recalls[np.isnan(recalls)] = 0.0
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # plot PR curve
    marker = 'x'
    markevery = 0.03
    plt.clf()
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    plt.plot(recalls, precisions, label='DeepRING', marker=marker, markevery=markevery)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall [%]")
    plt.ylabel("Precision [%]")
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
    plt.show()
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return precisions, recalls, f1s


def compute_AP(precisions, recalls):
    ap = 0.0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i-1])*precisions[i]
    return ap


def compute_AUC(precisions, recalls):
    auc = 0.0
    for i in range(len(precisions) - 1):
        auc += (recalls[i+1] - recalls[i])*(precisions[i+1] + precisions[i]) / 2.0
    return auc


if __name__ == '__main__':
    # settings
    bev_type = 'occ' # 'occ' or 'feat'
    method = 'RING' if bev_type == 'occ' else 'RING++'
    results_path = f'./results/nclt/test_2012-02-04_2012-03-17_20.0_5.0_10.0/revisit_5.0_10.0_15.0_20.0'
    revisit_thresholds = results_path.split('/')[-1].split('_')[1:]
    
    # load results
    precisions = np.loadtxt(f'{results_path}/precisions_{bev_type}.txt')
    recalls = np.loadtxt(f'{results_path}/recalls_{bev_type}.txt')
    f1s = np.loadtxt(f'{results_path}/f1s_{bev_type}.txt')

    # draw PR curve
    markers = ['o', 's', 'x', 'v', 'd', 'p', 'h']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig = plt.figure()

    for idx, revisit_threshold in enumerate(revisit_thresholds):
        # compute AP
        AP = compute_AP(precisions[idx], recalls[idx])
        print(f'AP: {AP}')
        
        # compute AUC
        AUC = compute_AUC(precisions[idx], recalls[idx])
        print(f'AUC: {AUC}')
        
        # plot
        plt.plot(recalls[idx], precisions[idx], marker = markers[idx], color = colors[idx], linewidth = 2, label = f'{method}({revisit_threshold}m)')
    
    plt.legend(loc = 'lower left')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
    # plot minor gridlines
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--', alpha=0.2)
    plt.grid(which='major', linestyle='-', alpha=0.5)
    # convert major tick lines from out direction to in direction
    plt.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    # remove minor tick lines and labels
    plt.tick_params(axis='both', which='minor', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.xlabel('Recall [%]')
    plt.ylabel('Precision [%]')
    plt.savefig(f'{results_path}/precision_recall_curve_{bev_type}.pdf', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()        