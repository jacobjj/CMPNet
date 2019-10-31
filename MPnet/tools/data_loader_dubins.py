"""A class to load the point cloud data and target points in pixels """
import os.path as osp
import numpy as np
import dubins

import torch
import torch.utils.data

from bc_gym_planning_env.envs.mini_env import RandomMiniEnv


def normalize_angle(z):
    """
    A function to wrap around -1 and 1
    """
    return (z + np.pi) % (2 * np.pi) - np.pi


def primitive2word(primitive_length, primitive_type, d):
    s = np.zeros((3, 3))
    scale = [d, d, 1]
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
        if i != 2:
            s[i, primitive2word_dict[row][i]] = normalize_angle(
                length / scale[primitive2word_dict[row][i]])
        else:
            s[i, primitive2word_dict[row][i]] = length / scale[
                primitive2word_dict[row][i]]
    return s


class DubinsDataset(torch.utils.data.Dataset):
    def __init__(self, folder_loc):
        self.folder_loc = folder_loc

    def __len__(self):
        return 4000

    def __getitem__(self, idx):
        env = RandomMiniEnv(
            turn_off_obstacles=False,
            draw_new_turn_on_reset=False,
            iteration_timeout=1200,
            seed=idx,
            goal_spat_dist=0.05,
        )
        obs = env.reset()
        costmap = obs.costmap
        costmap_data = costmap.get_data()
        costmap_dim = costmap_data.shape
        traj = np.load(osp.join(self.folder_loc, 'traj_{}.npy'.format(idx)))
        samples = traj.shape[0] - 1
        pixel_ind = [
            costmap.world_to_pixel(coordinate[:2]) for coordinate in traj[:-1]
        ]
        obs = np.ones((samples, costmap_dim[0] * 2, costmap_dim[1] * 2))
        inputs = np.zeros((samples, 6))
        targets = np.zeros((samples, 9))
        for j, _ in enumerate(traj[:-1]):
            x_0, y_0 = costmap_dim[1] - pixel_ind[j][1], costmap_dim[
                0] - pixel_ind[j][0]
            obs[j, x_0:x_0 + costmap_dim[1], y_0:y_0 +
                costmap_dim[0]] = costmap_data / 254
            inputs[j, :] = np.concatenate((traj[j], traj[-1]))
            path = dubins.shortest_path(tuple(traj[j]), tuple(traj[j + 1]),
                                        0.6)
            primitive_length = [path.segment_length(i) for i in range(3)]
            primitive_type = path.path_type()
            s = primitive2word(primitive_length, primitive_type, d=0.6)
            targets[j, :] = s.ravel()
        return {'obs': obs, 'inputs': inputs, 'targets': targets}


class DubinsIterDataset(torch.utils.data.IterableDataset):
    def __init__(self, folder_loc, seeds):
        self.folder_loc = folder_loc
        self.seeds = seeds

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.seeds)
        else:
            per_worker = int(len(self.seeds) // worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.seeds))

        return iter(self.GetItem(s) for s in self.seeds[iter_start:iter_end])

    def GetItem(self, idx):
        env = RandomMiniEnv(
            turn_off_obstacles=False,
            draw_new_turn_on_reset=False,
            iteration_timeout=1200,
            seed=idx,
            goal_spat_dist=0.05,
        )
        observation = env.reset()
        costmap = observation.costmap
        costmap_data = costmap.get_data()
        costmap_dim = costmap_data.shape
        traj = np.load(osp.join(self.folder_loc, 'traj_{}.npy'.format(idx)))
        samples = traj.shape[0] - 1
        pixel_ind = [
            costmap.world_to_pixel(coordinate[:2]) for coordinate in traj[:-1]
        ]
        for index in pixel_ind:
            if index[0] < 0 or index[1] < 0 or index[0]>costmap_dim[0] or index[1]>costmap_dim[1]:
                return {'obs': [], 'inputs': [], 'targets': []}
        height = int(costmap_dim[0] / 3) * 2
        width = int(costmap_dim[1] / 3) * 2
        obs = np.ones((samples, 1, height, width))
        inputs = np.zeros((samples, 6))
        targets = np.zeros((samples, 9))
        for j, _ in enumerate(traj[:-1]):
            x_0, y_0 = costmap_dim[1] - pixel_ind[j][1], costmap_dim[
                0] - pixel_ind[j][0]
            full_obs = np.ones((costmap_dim[0] * 2, costmap_dim[1] * 2))
            try:
                full_obs[x_0:x_0 + costmap_dim[1], y_0:y_0 +
                     costmap_dim[0]] = costmap_data / 254
            except:
                import pdb; pdb.set_trace()
            obs[j, 0, :, :] = full_obs[::3, ::3]
            inputs[j, :] = np.concatenate((traj[j], traj[-1]))
            path = dubins.shortest_path(tuple(traj[j]), tuple(traj[j + 1]),
                                        0.6)
            primitive_length = [path.segment_length(i) for i in range(3)]
            primitive_type = path.path_type()
            s = primitive2word(primitive_length, primitive_type, d=0.6)
            targets[j, :] = s.ravel()
        return {'obs': obs, 'inputs': inputs, 'targets': targets}
