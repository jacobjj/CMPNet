import torch
import numpy as np
from mpnet import MPnetBase
import copy


class MPNetPlan(MPnetBase):
    def __init__(self, modelFile, steerTo, stepSz=0.01, **kwargs):
        """
        A class that plans a path for a given environment and observation
        """
        super().__init__(self, **kwargs)

        self.stepSz = stepSz
        self.steerTo = steerTo
        # Load Model file
        self.load_network_parameters(modelFile)

    def getPath(self, IsInCollision, start, goal, obs):
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
        path = [
            torch.from_numpy(start).type(torch.FloatTensor),
            torch.from_numpy(goal).type(torch.FloatTensor)
        ]

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

            for j, _ in enumerate(path[:-1]):
                steer = self.steerTo(path[marker], path[marker + 1],
                                     IsInCollision)
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
                                              maxPoints=50)

                # Remove points that are in collision
                if len(miniPath) > 2:
                    miniPath = self.removeCollision(miniPath, IsInCollision)
                    # append miniPath to path
                    for i, point in miniPath:
                        path.insert(marker + i, point)

            # lazy vertex contraction
            path = self.lvc(path, IsInCollision)
            # The safety check that tests if the path generated is feasible or not.
            if self.feasibility_check(path, IsInCollision):
                print('feasible, ok!')
                return path

        return []

    def neuralPlan(self, start, goal, obs, IsInCollision, maxPoints):
        """
        Generate a path using MPnet
        """
        pA = [start]
        pB = [goal]
        noPath = [copy.copy(start), copy.copy(goal)]
        tree = 0
        for _ in range(maxPoints):
            if tree == 0:
                network_input = np.concatenate((start, goal), axis=1)
                tInput = self.format_input(obs, network_input)
                start = self.mpNet(tInput).squeeze(0).data.cpu()
                start = self.denormalize(start)
                pA.append(start)
                tree = 1
            else:
                network_input = np.concatenate((goal, start), axis=1)
                tInput = self.format_input(obs, network_input)
                goal = self.mpNet(tInput).squeeze(0).data.cpu()
                goal = self.denormalize(goal)
                pB.append(goal)
                tree = 0
            target_reached = self.steerTo(start, goal, IsInCollision)
            if target_reached:
                return pA.extend(reversed(pB)), 0
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
            if not IsInCollision(path[i].numpy()):
                new_path.append(path[i])
        return new_path
