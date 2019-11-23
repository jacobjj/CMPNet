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
    # s = 304299
    s = 0
    network_param = {
        'normalize': normalize,
        'denormalize': unnormalize,
        'encoderInputDim': [1, 122, 122],
        'encoderOutputDim': 128,
        'worldSize': [2.75, 2.75, np.pi],
        'AE': voxelNet,
        'MLP': model.MLP,
        'modelPath': 'data/MPnet_tricycle/points/Adam_lr_3_minus_4_centerObs_relativeTarget'
    }

    MPNetPlan_obj = MPNetPlan(
        modelFile=
        'data/MPnet_tricycle/points/Adam_lr_3_minus_4_centerObs_relativeTarget/mpnet_epoch_19.pkl',
        steerTo=steerTo,
        **network_param,
    )

    success = 0
    obstacle_path = 0
    for s in range(0,100):
        env = RandomMiniEnv(draw_new_turn_on_reset=False,
                            seed=s,
                            goal_spat_dist=0.05)
        observation = env.reset()
        costmap = observation.costmap
        robot = env._env._robot
        IsInCollision_env = lambda pose: pose_collides(
            pose[0],
            pose[1],
            pose[2],
            robot,
            costmap,
        )

        start_node = env._env._state.pose
        goal_node = env._env._state.original_path[-1]
        # Shifted costmap
        observation = env.reset()
        costmap = observation.costmap
        pointcloud = generate_point_cloud(s)
        # traj = np.load('data/dubinsCar/traj/traj_{}.npy'.format(s))
        # print("Start : {}, trajectory :{}".format(traj[0], traj[-1]))
        path = MPNetPlan_obj.getPath(
            IsInCollision_env,
            start_node,
            goal_node,
            costmap,
            pointcloud,
        )

        if len(path)==2:
            success += 1
            continue

        render = False
        if path:
            success += 1
            obstacle_path +=1
        if path and render:
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

            plt.pause(1)
            plt.close()

    print("Successful Paths : {}, Successful MPnet Paths : {} ".format(success, obstacle_path))
