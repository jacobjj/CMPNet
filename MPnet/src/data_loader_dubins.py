import torch
import argparse
import numpy as np
import re
import os
import os.path as osp
import dubins

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


"""
Define the word as follows:
index 0 - RightTurn
index 1 - LeftTurn
index 2 - Straight
Each row has the following representation
LSL = 0
LSR = 1
RSL = 2
RSR = 3
RLR = 4
LRL = 5
"""


def primitive2word(primitive_length, primitive_type, d):
    s = np.zeros((3, 3))
    scale = [d,d,1]
    primitive2word_dict = np.array([
        [1, 2, 1],
        [1, 2, 0],
        [0, 2, 1],
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 1],
    ])
    for i, length in enumerate(primitive_length):
        row = primitive_type
        s[i, primitive2word_dict[row][i]] = length/scale[primitive2word_dict[row][i]]
    return s


def load_dataset_voxel(N=10000, NP=1, folder_loc=None):
    """
    A function to load dataset for the BC environment in the same format as the other data loader functions
    : param N : Number of environments to load
    : param NP : Number of paths per environment
    : return array(N*NP,28000),array(N*NP,3),array(N*NP,3): point-cloud,inputs and target nodes
    """
    # TODO: Count number of possible samples that can be generated
    numSamples = N * NP

    inputs = np.zeros((numSamples, 6))
    targets = np.zeros((numSamples, 9))
    obs = np.zeros((numSamples, 1, 61, 61))
    i = 0
    done = False
    # Load data
    obsFolder = osp.join(folder_loc, 'obs_voxel')
    trajFolder = osp.join(folder_loc, 'traj')
    seeds = []

    for entry in os.listdir(trajFolder):
        if '.npy' in entry:
            s = int(re.findall(r'\d+', entry)[0])
            seeds.append(s)

    if not seeds:
        raise ValueError("{} - Not a valid folder".format(trajFolder))
    # Load point cloud, points and target information
    for s in seeds:
        obs_pc = np.load(osp.join(obsFolder, 'voxel_{}.npy'.format(s)))
        traj = np.load(osp.join(trajFolder, 'traj_{}.npy'.format(s)))
        for j, _ in enumerate(traj[:-1]):
            obs[i, ...] = np.expand_dims(obs_pc, axis=0)
            # Current node and goal location
            inputs[i, :] = np.concatenate((traj[j], traj[-1]))
            path = dubins.shortest_path(tuple(traj[j]), tuple(traj[j + 1]),
                                        0.6)
            primitive_length = [path.segment_length(i) for i in range(3)]
            primitive_type = path.path_type()
            s = primitive2word(primitive_length, primitive_type,d=0.6)
            targets[i, :] = np.ravel(s)
            i += 1
            if i == numSamples:
                done = True
                break
        if done:
            break
    if not done:
        print("Not enough samples")
        return obs[:i, ...], inputs[:i, ...], targets[:i, ...]

    return obs, inputs, targets
