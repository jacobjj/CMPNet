import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable
import math
import time
from plan_general import neural_replan
from plan_dubins import steerTo, DEFAULT_STEP, feasibility_check, lvc, generate_point_cloud
from bc_gym_planning_env.envs.mini_env import RandomMiniEnv

get_numpy = lambda x: x.data.cpu().numpy()


def eval_tasks(mpNet,
               test_data,
               filename,
               IsInCollision,
               normalize_func=lambda x: x,
               unnormalize_func=lambda x: x,
               time_flag=False):
    seeds = test_data
    fes_env = []  # list of list
    valid_env = np.ones((1, len(test_data)))
    time_env = []
    time_total = []
    # Going through each environment
    # Set this to seed value
    for i, s in enumerate(seeds):
        time_path = []
        fes_path = []  # 1 for feasible, 0 for not feasible
        fp = 0  # indicator for feasibility
        env = RandomMiniEnv(turn_off_obstacles=False,
                            draw_new_turn_on_reset=False,
                            iteration_timeout=12000,
                            seed=s,
                            goal_spat_dist=0.05)
        start = env._env._state.pose
        goal = env._env._state.original_path[-1]
        obc = dict(costmap=env._env._state.costmap, robot=env._env._robot)
        # Generate point cloud data and convert it to 1D array
        obs = generate_point_cloud(env._env.serialize()['costmap']).ravel()
        obs = torch.from_numpy(obs)
        time0 = time.time()
        time_norm = 0.

        print("Environment: {}".format(s))

        # Initialization of MPnet inputs
        path = [
            torch.from_numpy(start).type(torch.FloatTensor),
            torch.from_numpy(goal).type(torch.FloatTensor)
        ]
        # The step size for checking collision along a path
        step_sz = DEFAULT_STEP
        # Number of re-planning steps taken.
        MAX_NEURAL_REPLAN = 6

        for t in range(MAX_NEURAL_REPLAN):
            # adaptive step size on replanning attempts
            if (t == 2):
                step_sz = step_sz / 2
            elif (t == 3):
                step_sz = step_sz / 2
            elif (t > 3):
                step_sz = step_sz / 2
            if time_flag:
                # neural_replan is used to both generate the initial paths, and replan between points that are in collision
                path, time_norm = neural_replan(mpNet,
                                                path,
                                                obc,
                                                obs,
                                                IsInCollision,
                                                normalize_func,
                                                unnormalize_func,
                                                t == 0,
                                                step_sz=step_sz,
                                                time_flag=time_flag,
                                                steerTo=steerTo)
            else:
                path = neural_replan(mpNet,
                                     path,
                                     obc,
                                     obs,
                                     IsInCollision,
                                     normalize_func,
                                     unnormalize_func,
                                     t == 0,
                                     step_sz=step_sz,
                                     time_flag=time_flag,
                                     steerTo=steerTo)
            # lazy vertex contraction
            path = lvc(path, obc, IsInCollision, step_sz=step_sz)
            # The safety check that tests if the path generated is feasible or not.
            if feasibility_check(path, obc, IsInCollision, step_sz=0.01):
                fp = 1
                print('feasible, ok!')
                break
        if fp:
            # only for successful paths
            time1 = time.time() - time0
            time1 -= time_norm
            time_path.append(time1)
            numpy_path = np.array([get_numpy(p) for p in path])
            np.save('data/mpnet/test_env/traj_{}.npy'.format(s), numpy_path)
            print('test time: %f' % (time1))
        fes_path.append(fp)
        time_env.append(time_path)
        time_total += time_path
        print('average test time up to now: %f' % (np.mean(time_total)))
        fes_env.append(fes_path)
        print('accuracy up to now: %f' % (np.sum(fes_env) / np.sum(valid_env)))
    if filename is not None:
        pickle.dump(time_env, open(filename, "wb"))
        #print(fp/tp)
    return np.array(fes_env), np.array(valid_env)
