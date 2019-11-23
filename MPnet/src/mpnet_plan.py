import torch
import numpy as np
from src.mpnet import MPnetBase
import copy
import dubins
import matplotlib.pyplot as plt
from tools.utils import word2primitive


def primitive2word(primitive_length, primitive_type):
    s = np.zeros((3, 3))
    primitive2word = np.array([
        [1, 2, 1],
        [1, 2, 0],
        [0, 2, 1],
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 1],
    ])
    for i, length in enumerate(primitive_length):
        s[i, primitive2word[primitive_type][i]] = length
    return s

def normalize_cost(z):
    """
    A function to wrap around -1 and 1
    """
    return (z + 1) % (2 ) - 1


def CenterRobot(costmap, pixel_ind):
    costmap_data = costmap.get_data()
    costmap_dim = costmap_data.shape
    full_obs = np.ones((costmap_dim[0] * 2, costmap_dim[1] * 2))
    x_0, y_0 = costmap_dim[1] - pixel_ind[1], costmap_dim[0] - pixel_ind[0]
    full_obs[x_0:x_0 + costmap_dim[1], y_0:y_0 +
             costmap_dim[0]] = costmap_data / 254
    full_obs = full_obs[::3, ::3]
    full_obs = torch.tensor(full_obs).unsqueeze(0)
    return full_obs


class MPNetPlan(MPnetBase):
    def __init__(self, modelFile, steerTo, stepSz=0.01, **kwargs):
        """
        A class that plans a path for a given environment and observation
        TODO : Pickle BaseClass, making it easier for loading and running models
        """
        super().__init__(**kwargs)

        self.stepSz = stepSz
        self.steerTo = steerTo
        # Load Model file
        self.load_network_parameters(modelFile)

    def plotMainPath(self, ax, path, pointCloud):
        """
        A function to plot the point cloud information and path
        """
        ax.scatter(pointCloud[:, 0], pointCloud[:, 1], color='c')
        if path:
            ax.scatter(path[0][0], path[0][1], marker='o', color='r')
            ax.scatter(path[-1][0], path[-1][1], marker='o', color='g')
            self.plotPoints(ax, path[1:-1], color='m')

    def plotPoints(self, ax, path, color='r'):
        """
        Plot the points in path
        """
        for p in path:
            ax.scatter(p[0], p[1], marker='x', color=color)
            ax.arrow(p[0],
                     p[1],
                     np.cos(p[2]),
                     np.sin(p[2]),
                     head_width=0.1,
                     head_length=0.3)

    def getDubinsTraj(self, path):
        """
        Return the dubin trajectory of the current path
        """
        traj = []
        for i, _ in enumerate(path[:-1]):
            p = dubins.shortest_path(tuple(path[i]), tuple(path[i + 1]), d=0.6)
            configs, _ = p.sample_many(0.1)
            traj.extend(configs)
        return traj

    def plotDubinPath(self, ax, path):
        traj = self.getDubinsTraj(path)
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1])

    def getPath(self, IsInCollision, start, goal, costmap, pointCloud=None):
        """
        Generate a plan using the MPnet
        : param IsInCollision :
        : param start numpy.array: start position of the robot
        : param goal numpy.array : goal position of the robot
        : param costmap : A costmap object from bc_gym_planning_env
        : return path list: key points to generate a path
        """
        render = False
        start = torch.tensor(start, dtype=torch.float)
        goal = torch.tensor(goal, dtype=torch.float)
        path = [start, goal]
        if render:
            fig, (axMain, axMini) = plt.subplots(1, 2)
            figParam = {'axis': axMini, 'pointCloud': pointCloud}
            for ax in (axMain, axMini):
                ax.set_xlim([-5, 5])
                ax.set_ylim([-5, 5])
                self.plotMainPath(axMain, path, pointCloud)
        else:
            figParam = None
        # Number of re-planning steps taken.
        MAX_NEURAL_REPLAN = 5
        step_sz = 0.01
        for t in range(MAX_NEURAL_REPLAN):
            # adaptive step size on replanning attempts
            if (t == 2):
                step_sz = step_sz / 2
            elif (t == 3):
                step_sz = step_sz / 2
            elif (t > 3):
                step_sz = step_sz / 2

            print("Plan {}".format(t))
            for j, _ in enumerate(path[:-1]):
                steer = self.steerTo(path[j], path[j + 1], IsInCollision)
                if not steer:
                    marker = j
                    # Remove the two points
                    startNode = path.pop(marker)
                    goalNode = path.pop(marker)
                    break

            # Generate a plan
            if not steer:
                miniPath, _ = self.neuralPlan(startNode,
                                              goalNode,
                                              costmap,
                                              IsInCollision,
                                              maxPoints=2,
                                              figParam=figParam)
                # import pdb;pdb.set_trace()

                print("Remove points in Collision")
                # Remove points that are in collision
                print(miniPath)
                if len(miniPath) > 2:
                    miniPath = self.removeCollision(miniPath, IsInCollision)
                print(miniPath)
                if render:
                    axMini.clear()
                    self.plotMainPath(axMini, [], pointCloud)
                    self.plotPoints(axMini, miniPath)
                    for ax in (axMain, axMini):
                        ax.set_xlim([-5, 5])
                        ax.set_ylim([-5, 5])
                    plt.pause(1)

                # append miniPath to path
                for i, point in enumerate(miniPath):
                    path.insert(marker + i, point.clone())
                if render:
                    axMain.clear()
                    self.plotMainPath(axMain, path, pointCloud)
                    for ax in (axMain, axMini):
                        ax.set_xlim([-5, 5])
                        ax.set_ylim([-5, 5])
                    plt.pause(2)

            # lazy vertex contraction
            print("LVC")
            path = self.lvc(path, IsInCollision)
            # The safety check that tests if the path generated is feasible or not.
            if self.feasibility_check(path, IsInCollision):
                print('feasible, ok!')
                plt.close()
                return path

        plt.close()
        return []

    def formatInput(self, v):
        if len(v.shape) == 1:
            return np.expand_dims(v, 0)

    def neuralPlan(self,
                   start,
                   goal,
                   costmap,
                   IsInCollision,
                   maxPoints,
                   figParam=None):
        """
        Generate a path using MPnet
        """
        # start = self.formatInput(start)
        # goal = self.formatInput(goal)
        noPath = [start.clone(), goal.clone()]
        pA = [start.clone()]
        tree = 0
        center_obs = CenterRobot(costmap,
                                 costmap.world_to_pixel(start[:2].numpy()))
        # traj = np.load('data/dubinsCar/traj/traj_304299.npy')
        # traj = self.normalize(torch.tensor(traj), self.worldSize)
        # traj = traj.numpy()
        with torch.no_grad():
            for _ in range(maxPoints):
                network_input = torch.cat((start, goal),dim=0).reshape(1,-1)
                tobs, tInput = self.format_input(center_obs.unsqueeze(0), network_input)
                # word = self.mpNet(tInput, tobs).squeeze().data.cpu()
                # temp = word2primitive(word, start, 0.6)
                temp = self.mpNet(tInput, tobs).squeeze().data.cpu()
                # print("Estimate Loss : {} ".format(self.EstimateLoss(temp.numpy(), traj[1])))
                temp = self.denormalize(temp, self.worldSize)
                # print(temp)
                # print("sampled Point : {}".format(temp))
                pixel_ind = costmap.world_to_pixel(temp[:2].numpy())
                if pixel_ind[0]<0 or pixel_ind[1]<0:
                    continue
                if pixel_ind[0]>183 or pixel_ind[1]>183:
                    continue
                start = temp
                pA.append(start.clone())

                center_obs = CenterRobot(
                    costmap, costmap.world_to_pixel(start[:2].numpy()))

                target_reached = self.steerTo(start.squeeze(), goal.squeeze(),
                                          IsInCollision)
                if target_reached:
                    break
        
        if figParam is not None:
            ax = figParam['axis']
            pointCloud = figParam['pointCloud']
            sampledPoints = pA
            self.plotPoints(ax, sampledPoints)
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
            plt.pause(1)
        if target_reached:
            pA.append(goal.clone())
            return pA, 0
        return noPath, 0

    def lvc(self, path, IsInCollision):
        """
        Lazy vertex contraction
        """
        # lazy vertex contraction
        for i in range(0, len(path) - 1):
            for j in range(len(path) - 1, i + 1, -1):
                ind = 0
                ind = self.steerTo(path[i],
                                   path[j],
                                   IsInCollision,
                                   step_sz=self.stepSz)
                if ind == 1:
                    pc = []
                    for k in range(0, i + 1):
                        pc.append(path[k].clone())
                    for k in range(j, len(path)):
                        pc.append(path[k].clone())
                    return self.lvc(pc, IsInCollision)
        return path

    def feasibility_check(self, path, IsInCollision):
        """
         checks the feasibility of entire path including the path edge by checking for each adjacent vertices
        """
        for i, _ in enumerate(path[:-1]):
            if not self.steerTo(
                    path[i], path[i + 1], IsInCollision, step_sz=self.stepSz):
                # collision occurs from adjacent vertices
                return 0
        return 1

    def removeCollision(self, path, IsInCollision):
        """
        Rule out nodes that are already in collision
        """
        new_path = []

        for i in range(0, len(path)):
            if not IsInCollision(path[i]):
                new_path.append(path[i].clone())
        return new_path

    def EstimateLoss(self, pred, truth):
        loss = pred - truth
        loss[:2] = loss[:2]**2
        loss[2] = normalize_cost(loss[2])**2
        return np.sum(loss)
