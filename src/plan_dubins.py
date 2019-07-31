'''
requires the necessary files to test for collision
'''
from bc_gym_planning_env.envs.base.env import pose_collides
from bc_gym_planning_env.utilities.coordinate_transformations import normalize_angle
from bc_gym_planning_env.utilities.coordinate_transformations import pixel_to_world
import dubins
import numpy as np

DEFAULT_STEP = 0.01


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
    d = 0.964
    path = dubins.shortest_path(start, end, d)
    traj, _ = path.sample_many(step_sz)
    traj = np.array(traj)

    for p in traj:
        if IsInCollision(p, obc):
            return 0
    return 1


def feasibility_check(path, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # checks the feasibility of entire path including the path edges
    # by checking for each adjacent vertices
    for i in range(0, len(path) - 1):
        if not steerTo(
                path[i], path[i + 1], obc, IsInCollision, step_sz=step_sz):
            # collision occurs from adjacent vertices
            return 0
    return 1


def lvc(path, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # lazy vertex contraction
    for i in range(0, len(path) - 1):
        for j in range(len(path) - 1, i + 1, -1):
            ind = 0
            ind = steerTo(path[i],
                          path[j],
                          obc,
                          IsInCollision,
                          step_sz=step_sz)
            if ind == 1:
                pc = []
                for k in range(0, i + 1):
                    pc.append(path[k])
                for k in range(j, len(path)):
                    pc.append(path[k])
                return lvc(pc, obc, IsInCollision, step_sz=step_sz)
    return path


def generate_point_cloud(costmap):
    shape = costmap['data'].shape
    index = []
    assert shape[0] == shape[
        1], "Sampling not implemented for costmap with different dimensions"
    while len(index) <= 1400:
        sample_ind = np.random.randint(0, shape[0], size=(500, 2))

        useful_index = costmap['data'][sample_ind[:, 0],
                                       sample_ind[:, 1]] == 254
        sel_ind = sample_ind[useful_index, :]
        index.extend(sel_ind)

    # NOTE: get_physical_coords_from_drawing did not correspond to world coordintates as observed from rendering
    point_cloud = pixel_to_world(index[:1400], costmap['origin'],
                                 costmap['resolution'])
    # Rotate point-cloud, since costmap was flipped
    point_cloud[:, 0], point_cloud[:, 1] = point_cloud[:,
                                                       1], point_cloud[:,
                                                                       0].copy(
                                                                       )
    return point_cloud
