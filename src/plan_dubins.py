'''
requires the necessary files to test for collision
'''
from bc_gym_planning_env.envs.base.env import pose_collides
from bc_gym_planning_env.utilities.coordinate_transformations import normalize_angle
import dubins
import numpy as np


def IsInCollision(pose, obc):
    """
    Function to check if the given pose is in collision for the given costmap, and robot model.
    :param pose(tuple): the robot pose, used to check collision
    :param obc(dictionary): contains the costmap and robot state objects
    :returns True(bool) if the robot  is in collision, else False
    """

    return pose_collides(
        pose[0],
        pose[1],
        pose[2],
        obc['robot'],
        obc['costmap'],
    )


def steerTo(start, end, obc, IsInCollision, step_sz=0.01):
    """
    A function to generate a dubins curve between start and end, and test if the points in the path are in  collision.
    :param start: the start position of the robot
    :param end: the goal position of the robot
    :param obc: a dictionary containing the costmap and robot state objects
    :param IsInCollision: Collision check function
    :param step_sz: The step size for generating the points on the path
    :return (0/1): 0 if the path is possible, else 0
    """

    angle_diff = normalize_angle(start[2] - end[2])
    if abs(angle_diff) > np.pi / 2:
        return 0
    # NOTE: This will only work for Tricycle model.
    d = 0.03
    path = dubins.shortest_path(start, end, d)
    traj, _ = path.sample_many(0.01)
    traj = np.array(traj)

    for p in traj:
        if IsInCollision(p, obc):
            return 0
    return 1
