import numpy as np
from bc_gym_planning_env.envs.mini_env import RandomMiniEnv
from bc_gym_planning_env.envs.base.env import pose_collides

from src.utility_r2d import normalize, unnormalize
import Model.AE.voxel_AE as voxelNet
import Model.model as model
from src.mpnet_plan import MPNetPlan
from src.plan_dubins import steerTo
from BC.utils.misc import generateVoxelData
from BC.scripts.point_cloud import generate_point_cloud

import dubins


def steerTo_env(start, goal, IsInCollision):
    steerTo(start, goal, IsInCollision)


import matplotlib.pyplot as plt

if __name__ == "__main__":
    s = 89
    env = RandomMiniEnv(draw_new_turn_on_reset=False,
                        seed=s,
                        goal_spat_dist=0.05)
    costmap = env._env._state.costmap
    robot = env._env._robot
    IsInCollision_env = lambda pose: pose_collides(
        pose[0],
        pose[1],
        pose[2],
        robot,
        costmap,
    )

    network_param = {
        'normalize': normalize,
        'denormalize': unnormalize,
        'encoderInputDim': [1, 122, 122],
        'encoderOutputDim': 128,
        'worldSize': [2.75, 2.75, np.pi],
        'AE': voxelNet,
        'MLP': model.MLP,
        'modelPath': 'data/MPnet_tricycle/points/Adam_lr_3_minus_4_centerObs'
    }

    MPNetPlan_obj = MPNetPlan(
        modelFile=
        'data/MPnet_tricycle/points/Adam_lr_3_minus_4_centerObs/mpnet_epoch_99.pkl',
        steerTo=steerTo,
        **network_param,
    )
    start_node = env._env._state.pose
    goal_node = env._env._state.original_path[-1]
    # Shifted costmap
    observation = env.reset()
    costmap = observation.costmap
    pointcloud = generate_point_cloud(s)
    path = MPNetPlan_obj.getPath(
        IsInCollision_env,
        start_node,
        goal_node,
        costmap,
        pointcloud,
    )

    if path:
        plt.figure()
        plt.scatter(pointcloud[:, 0], pointcloud[:, 1])
        for p in path:
            plt.scatter(p[0], p[1], marker='x', color='r')
        for i, _ in enumerate(path[:-1]):
            p = dubins.shortest_path(tuple(path[i].numpy()),
                                     tuple(path[i + 1].numpy()), 0.6)
            config, _ = p.sample_many(0.1)
            config = np.array(config)
            plt.plot(config[:, 0], config[:, 1])

        plt.show()
