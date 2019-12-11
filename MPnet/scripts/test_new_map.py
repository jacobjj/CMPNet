import numpy as np
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D

from bc_gym_planning_env.envs.base.env import PlanEnv
from bc_gym_planning_env.envs.base.params import EnvParams

import cv2

if __name__=="__main__":
    data = cv2.imread('data/maps/ucsd_atkinson_6f_icra2019_fx_net.pgm', -1)
    resolution =  0.1
    origin = np.asarray([-57.750001, -41.950002])
    costmap = CostMap2D(
        data=np.asarray(data),
        resolution=resolution,
        origin=origin
    )
    env = PlanEnv(
        costmap=costmap,
        path=np.array([[0.,0.,0.],[10.,0.,0.]]),
        params=EnvParams()
    )