import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'font.weight': "bold"})
matplotlib.rcParams.update({'axes.labelweight': "bold"})


# plot cdf
def plot_cdf(errors, save_path, xlabel, ylabel, title):
    errors = np.sort(errors)
    y = np.arange(len(errors))/float(len(errors))
    plt.plot(errors, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


# convert numpy to dataframe
def data_to_df(data, label, method):
    num = len(data)
    data_new = np.empty((num, 3), dtype=object)

    for i in range(num):
        data_new[i, 0] = data[i]
        data_new[i, 1] = label
        data_new[i, 2] = method
        
    data_df = pd.DataFrame(data_new)
    data_df.columns = ['Error', 'Threshold', 'Method']
    
    return data_df


if __name__ == "__main__":
    # settings
    icp = False
    bev_type = 'occ' # 'occ' or 'feat'
    method = 'RING' if bev_type == 'occ' else 'RING++'
    results_path = f'./results/nclt/test_2012-02-04_2012-03-17_20.0_5.0_10.0/revisit_5.0_10.0_15.0_20.0'
    revisit_thresholds = results_path.split('/')[-1].split('_')[1:]
    quantiles = [0.25, 0.5, 0.75, 0.95]
    
    # load results
    if icp:
        rot_error = np.loadtxt(f'{results_path}/icp_rot_errors_{bev_type}.txt') 
        trans_error = np.loadtxt(f'{results_path}/icp_trans_errors_{bev_type}.txt') 
    else:
        rot_error = np.loadtxt(f'{results_path}/rot_errors_{bev_type}.txt') 
        trans_error = np.loadtxt(f'{results_path}/trans_errors_{bev_type}.txt')         

    # convert to dataframe
    df_rot_total = pd.DataFrame()
    df_trans_total = pd.DataFrame()
    for idx, revisit_threshold in enumerate(revisit_thresholds):
        rot_err = rot_error[idx]
        trans_err = trans_error[idx]
        df_rot = data_to_df(rot_err, label=revisit_threshold, method=method)
        df_trans = data_to_df(trans_err, label=revisit_threshold, method=method)
        df_rot_total = pd.concat([df_rot, df_rot_total])
        df_trans_total = pd.concat([df_trans, df_trans_total])
        
        rot_error_quantiles = np.quantile(rot_err, quantiles)
        trans_error_quantiles = np.quantile(trans_err, quantiles)
        print(f"Rotation error quantiles at {revisit_threshold} m: {rot_error_quantiles}")
        print(f"Translation error quantiles at {revisit_threshold} m: {trans_error_quantiles}")        
        rot_success = rot_err[rot_err<=5]
        trans_success = trans_err[trans_err<=2]
        num = 0
        for j in range(len(rot_err)):
            if rot_err[j] <= 5 and trans_err[j] <= 2:
                num += 1
        print(f"Rotation success rate at {revisit_threshold} m: ", len(rot_success)/len(rot_err))
        print(f"Translation success rate at {revisit_threshold} m: ", len(trans_success)/len(trans_err))
        print(f"Pose success rate at {revisit_threshold} m: ", num/len(rot_err))

    markers = ['o', 's', 'x', 'v', 'd', 'p', 'h']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # draw boxplot for rotation and translation error
    fig, axs = plt.subplots(1,2, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    my_pal = {"RING": colors[0], "RING++": colors[1]}
    sns.boxplot(data=df_rot_total, x='Threshold', y='Error', hue='Method', width=0.5, fliersize=2, linewidth=1.5, ax=axs[0], palette=my_pal)
    axs[0].set_yscale('log')
    axs[0].set_ylim(1e-1, max(df_rot_total['Error'].max(), df_rot_total['Error'].max()))
    axs[0].set_xlabel("Revisit Threshold (m)")
    axs[0].set_ylabel("Rotation Error (deg)")
    axs[0].legend([],[], frameon=False)

    sns.boxplot(data=df_trans_total, x='Threshold', y='Error', hue='Method', width=0.5, fliersize=2, linewidth=1.5, ax=axs[1], palette=my_pal)
    axs[1].set_ylim([0, 20])
    axs[1].set_xlabel("Revisit Threshold (m)")
    axs[1].set_ylabel("Translation Error (m)")
    axs[1].legend([],[], frameon=False)
    plt.legend(loc='upper right')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.savefig(os.path.join(results_path, f"pose_estimation_error_{bev_type}.pdf"), dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
