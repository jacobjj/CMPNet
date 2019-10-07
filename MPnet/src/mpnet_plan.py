import torch
import numpy as np
from src.mpnet import MPnetBase
import copy
import dubins
import matplotlib.pyplot as plt

# def plot_arrow(x, y, yaw):


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

    def getPath(self, IsInCollision, start, goal, obs, pointCloud=None):
        """
        Generate a plan using the MPnet
        : param IsInCollision :
        : param start numpy.array: start position of the robot
        : param goal numpy.array : goal position of the robot
        : param obs numpy.array: obstacle representation of the robot
        : return path list: key points to generate a path
        """
        obs = torch.from_numpy(obs)
        # NOTE: Do we need to convert them here or do we need to convert them only while  executing in the network
        # path = [
        #     torch.from_numpy(start).type(torch.FloatTensor),
        #     torch.from_numpy(goal).type(torch.FloatTensor)
        # ]
        path = [start, goal]
        fig, (axMain, axMini) = plt.subplots(1, 2)
        figParam = {'axis': axMini, 'pointCloud': pointCloud}
        for ax in (axMain, axMini):
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
        self.plotMainPath(axMain, path, pointCloud)
        # Number of re-planning steps taken.
        MAX_NEURAL_REPLAN = 6
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
                                              obs,
                                              IsInCollision,
                                              maxPoints=50,
                                              figParam=figParam)
                # import pdb;pdb.set_trace()

                print("Remove points in Collision")
                # Remove points that are in collision
                if len(miniPath) > 2:
                    miniPath = self.removeCollision(miniPath, IsInCollision)

                axMini.clear()
                self.plotPoints(axMini, miniPath)
                self.plotMainPath(axMini, [], pointCloud)
                for ax in (axMain, axMini):
                    ax.set_xlim([-5, 5])
                    ax.set_ylim([-5, 5])
                plt.pause(1)

                # append miniPath to path
                for i, point in enumerate(miniPath):
                    path.insert(marker + i, point)

                axMain.clear()
                self.plotMainPath(axMain, path, pointCloud)
                for ax in (axMain, axMini):
                    ax.set_xlim([-5, 5])
                    ax.set_ylim([-5, 5])
                plt.pause(2)

            # lazy vertex contraction
            path = self.lvc(path, IsInCollision)
            # The safety check that tests if the path generated is feasible or not.
            if self.feasibility_check(path, IsInCollision):
                print('feasible, ok!')
                return path

        return []

    def formatInput(self, v):
        if len(v.shape) == 1:
            return np.expand_dims(v, 0)

    def neuralPlan(self,
                   start,
                   goal,
                   obs,
                   IsInCollision,
                   maxPoints,
                   figParam=None):
        """
        Generate a path using MPnet
        """
        # start = self.formatInput(start)
        # goal = self.formatInput(goal)
        pA = [start]
        pB = [goal]
        noPath = [copy.copy(start), copy.copy(goal)]
        tree = 0
        for _ in range(maxPoints):
            if tree == 0:
                network_input = np.concatenate((start, goal))
                network_input = self.formatInput(network_input)
                tobs, tInput = self.format_input(obs, network_input)
                # start = self.mpNet(tInput, tobs).squeeze().data.cpu()
                # NOTE: Changed to be compatible with policy
                start, _ = self.mpNet.sample(tobs, tInput)
                start = start.squeeze().data.cpu()
                start = self.denormalize(start, self.worldSize).numpy()
                pA.append(start)
                tree = 1
            else:
                network_input = np.concatenate((goal, start))
                network_input = self.formatInput(network_input)
                tobs, tInput = self.format_input(obs, network_input)
                # goal = self.mpNet(tInput, tobs).squeeze(0).data.cpu()
                goal, _ = self.mpNet.sample(tobs, tInput)
                goal = goal.squeeze().data.cpu()
                goal = self.denormalize(goal, self.worldSize).numpy()
                pB.append(goal)
                tree = 0

            target_reached = self.steerTo(start.squeeze(), goal.squeeze(),
                                          IsInCollision)
            if figParam is not None:
                ax = figParam['axis']
                pointCloud = figParam['pointCloud']
                sampledPoints = pA + pB
                self.plotPoints(ax, sampledPoints)
                ax.set_xlim([-5, 5])
                ax.set_ylim([-5, 5])
                plt.pause(1)
            if target_reached:
                pA.extend(reversed(pB))
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
                        pc.append(path[k])
                    for k in range(j, len(path)):
                        pc.append(path[k])
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
                new_path.append(path[i])
        return new_path
