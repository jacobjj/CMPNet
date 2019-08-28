import torch
from mpnet import MPnetBase


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

    def getPlan(self, IsInCollision, start, goal, obs):
        """
        Generate a plan using the MPnet
        : param IsInCollision :
        : param start numpy.array: start position of the robot
        : param goal numpy.array : goal position of the robot
        : param obs numpy.array: obstacle representation of the robot
        : return path list: key points to generate a path
        """
        obs = torch.from_numpy(obs)
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

            # neural_replan is used to both generate the initial paths, and replan between points that are in collision
            path, _ = neural_replan(self.mpNet,
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
            path = self.lvc(path, IsInCollision)
            # The safety check that tests if the path generated is feasible or not.
            if self.feasibility_check(path, IsInCollision):
                print('feasible, ok!')
                return path
        return []

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
        # checks the feasibility of entire path including the path edges
        # by checking for each adjacent vertices
        for i, _ in enumerate(path[:-1]):
            if not self.steerTo(
                    path[i], path[i + 1], IsInCollision, step_sz=self.stepSz):
                # collision occurs from adjacent vertices
                return 0
        return 1
