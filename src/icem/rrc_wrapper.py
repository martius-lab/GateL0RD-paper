import numpy as np
import random
import torch
from environments.remote_control_gym import RemoteControlGym
from mbrl.environments.abstract_environments import GroundTruthSupportEnv


class RobotRemoteControlEnv(GroundTruthSupportEnv, RemoteControlGym):

    """
    Interface of the Robot Remote Control gym to work with iCEM (https://github.com/martius-lab/iCEM)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.goal_state[2:] = np.copy(self.goal_pos[:])
        self.goal_mask = np.array([0.0, 0.0, 1.0, 1.0])
        self.goal_threshold *= 1.5

    def get_GT_state(self):
        rob_contr = np.array([0])
        if self.robot_controlled:
            rob_contr[0] = 1
        return np.concatenate((self.agent_pos.flat, self.robot_pos.flat, self.computer_pos.flat, self.goal_pos.flat, self.last_action.flat, rob_contr.flat))

    def seed(self, seed):
        self.r_seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def set_GT_state(self, state):
        self.agent_pos = np.copy(state[0:2])
        self.robot_pos = np.copy(state[2:4])
        self.robot_pos[0] += 2
        self.computer_pos = np.copy(state[4:6])
        self.goal_pos = np.copy(state[6:8])
        self.last_action = np.copy(state[8:10])
        self.robot_controlled = state[10] == 1

    def set_state_from_observation(self, observation):
        rob_contr = np.array([0])
        computer_pos = np.array([0.5, 0.5])
        goal_pos = np.array([2.5, -0.5])
        last_action = np.array([-0.1, 0])
        if self.robot_controlled:
            rob_contr[0] = 1
        state = np.concatenate((observation.flat, computer_pos.flat, goal_pos.flat, last_action.flat, rob_contr.flat))
        self.set_GT_state(state)


    def cost_fn(self, observation, action, next_obs):

        # Bring goal position into same frame of reference as robot in observations
        goal_robot_pos = torch.from_numpy((np.copy(self.goal_pos[:])))
        goal_robot_pos[0] -= 2

        # To batch
        goal_robot_pos_observations2 = goal_robot_pos.expand(observation.shape[0], observation.shape[1], 2)

        # Distance between robot and goal pos
        return torch.norm((next_obs[:, :, 2:] - goal_robot_pos_observations2), 2, 2)