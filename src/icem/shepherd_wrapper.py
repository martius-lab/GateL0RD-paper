import numpy as np
import random
import torch
from environments.shepherd_gym import ShepherdGym
from mbrl.environments.abstract_environments import GroundTruthSupportEnv


class SheepEnv(GroundTruthSupportEnv, ShepherdGym):
    """
    Interface of the shepherd gym to work with iCEM (https://github.com/martius-lab/iCEM)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.goal_state[4:6] = np.copy(self.cage_pos[:])
        self.goal_mask = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        self.cage_threshold *= 1.2
        self.lever_threshold *= 1.2

    def get_GT_state(self):
        stable_height = np.array([self.stable_height])
        cage_controlled = np.array([0])
        if self.cage_carried:
            cage_controlled[0] = 1.0
        sheep_hidden = np.array([0])
        if self.sheep_hidden:
            sheep_hidden[0] = 1
        sheep_in_cage = np.array([0, self.sheep_velocity])
        if self.sheep_in_cage:
            sheep_in_cage[0] = 1
        return np.concatenate((self.agent_pos.flat, self.cage_pos.flat, self.sheep_pos.flat, stable_height.flat, cage_controlled.flat, sheep_hidden.flat, sheep_in_cage.flat))


    def reset(self):
        o_t = super(SheepEnv, self).reset(reset_right=True)
        self.agent_pos[0] = 2.9
        self.cage_pos[0] = 2.9
        agent_pos_observation = self.pos_to_obs(self.agent_pos)
        cage_pos_observation = self.pos_to_obs(self.cage_pos)
        o_t[0:2] = agent_pos_observation[:]
        o_t[2:4] = cage_pos_observation[:]
        return  o_t


    def seed(self, seed):
        self.r_seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def set_GT_state(self, state):
        self.agent_pos = np.copy(state[0:2])
        self.sheep_pos = np.copy(state[2:4])
        self.sheep_pos[0] += 2
        self.lever_pos = np.copy(state[4:6])
        self.last_action = np.copy(state[8:10])
        self.sheep_in_cage = state[10] == 1

    def set_state_from_observation(self, observation):
        sheep_caged = np.array([0])
        lever_pos = np.array([0.5, 0.5])
        goal_pos = np.array([2.5, -0.5])
        last_action = np.array([-0.1, 0])
        if self.robot_controlled:
            sheep_caged[0] = 1
        state = np.concatenate((observation.flat, lever_pos.flat, goal_pos.flat, last_action.flat, sheep_caged.flat))
        self.set_GT_state(state)


    def cost_fn(self, observation, action, next_obs):

        # If sheep is behind wall or above, give a large cost
        sheep_not_in_field_cost = (next_obs[:, :, 5] > -0.1) * 2

        # Otherwise take the distacne between sheep and cage
        cage_pos = self.obs_to_pos_batch(next_obs[:, :, 2:4])
        sheep_pos = self.sheep_obs_to_pos_batch(next_obs[:, :, 4:6])
        dist_cage_to_sheep = torch.norm((cage_pos - sheep_pos), 2, 2)

        return (sheep_not_in_field_cost + dist_cage_to_sheep).clamp(0, 2)