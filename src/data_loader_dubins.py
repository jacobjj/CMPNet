import torch
import argparse
import numpy as np
import re
import os
import os.path as osp

# from bc_gym_planning_env.envs.mini_env import RandomMiniEnv
# from bc_gym_planning_env.envs.base.env import pose_collides
# from BC.scripts.rrt_star_path_planning import follow_trajectory, plot_path
# import dubins

# from bc_gym_planning_env.utilities.map_drawing_utils import get_physical_coords_from_drawing


def get_files(folder_loc):
    files = []
    for _, _, f in os.walk(folder_loc):
        for f_i in f:
            if '.npy' in f_i:
                files.append(f_i)

    return files


def load_dataset(N=10000, folder_loc=None):
    """
    A function to load dataset for the BC environments
    :param N: Number of points that needs to be loaded in the dataset.
    : return array(N,28000),array(N,3),array(N,3): point-cloud,inputs and target nodes
    """
    obs = np.zeros((N, 2800))
    inputs = np.zeros((N, 6))
    targets = np.zeros((N, 3))
    i = 0
    done = False
    #Load possible seeds:
    if not folder_loc:
        pc_folder = osp.join('data', 'point_cloud')
        prune_folder = osp.join('data', 'RRT_star_planning_prune')
    else:
        pc_folder = osp.join('data', 'point_cloud2')
        prune_folder = osp.join('data', 'RRT_star_planning_prune2')
    seeds = []
    for _, _, f in os.walk(prune_folder):
        for f_i in f:
            if '.npy' in f_i:
                s = int(re.findall(r'\d+', f_i)[0])
                seeds.append(s)

    if not seeds:
        print("Check folder location")
        # Load point cloud, points and target information
    for s in seeds:
        obs_pc = np.load(osp.join(pc_folder, 'pc_{}.npy'.format(s)))
        traj = np.load(osp.join(prune_folder, 'traj_{}.npy'.format(s)))
        for j, _ in enumerate(traj[:-1]):
            obs[i, :] = obs_pc.ravel()
            # Current node and goal location
            inputs[i, :] = np.concatenate((traj[j], traj[-1]))
            targets[i, :] = traj[j + 1]
            i += 1
            if i == N:
                done = True
                break
        if done:
            break
    if not done:
        print("Not enough samples")

    return obs, inputs, targets
