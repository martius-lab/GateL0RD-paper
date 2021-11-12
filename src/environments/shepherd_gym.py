"""
Control task where an agent can catch a sheep by previously placing a cage and then pulling a lever
to open a cage and lure the sheep into the cage. The challenge of this scenario is that the sheep is only
briefly visible before it's occluded. Thus, for planning to catch it, the sheep's position needs to be
memorized.

The scenario is implemented in the OpenAi Gym interface (gym.openai.com)

"""

import math
import gym
from gym import spaces, logger
import  remote_control_gym_rendering as rendering
import numpy as np
import pyglet
import random
from pyglet.gl import *
from pyglet.image.codecs.png import PNGImageDecoder
import os
import torch

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'


class ShepherdGym(gym.Env):

    def __init__(self, r_seed):
        """
        Gym task for capturing a sheep

        Observation (7 dimensional):
        - Position of agent (x, y)
        - Position of cage (x, y)
        - Position of sheep (x, y)
        - Height of wall

        Action (2-dimensional):
        - Movement of agent (x, y)
        - Grasp/Drop off action

        :param r_seed: random seed
        """
        np.random.seed(r_seed)
        random.seed(r_seed)
        self.agent_pos = (np.random.rand(2) - 0.5) * 2
        self.agent_pos[0] = (self.agent_pos[0] - 0.5) * 4 + 1
        self.agent_pos[1] = (self.agent_pos[1] - 1.0)

        self.cage_pos = self.agent_pos + (np.random.rand(2) - 0.5) * 0.1

        sheep_pos_x_factor = 1.0
        self.sheep_pos = (np.random.rand(2) - 0.5) * sheep_pos_x_factor + np.array([2.0, 0.0], dtype=np.float64)
        self.sheep_pos[1] = self.sheep_pos[1] * 0.01 + 0.95

        self.lever_pos = np.array([-0.85, 0.0], dtype=np.float64)
        self.r_seed = r_seed

        self.agent_pos_upper_limits = np.array([2.95, -0.05], dtype=np.float64)
        self.agent_pos_lower_limits = np.array([-0.95, -0.95], dtype=np.float64)
        self.sheep_pos_upper_limits = np.array([2.95, 0.95], dtype=np.float64)
        self.sheep_pos_lower_limits = np.array([1.05, -1.05], dtype=np.float64)
        self.sheep_pos_lower_limits_closed = np.array([1.05, 0.0], dtype=np.float64)
        self.position_limits = np.array([0.95, 0.95, -0.95, -0.95, 2.95, 0.95, 1.05, -0.95], dtype=np.float64)
        action_limits = np.array([1, 1, 1])
        self.action_space = spaces.Box(-1 * action_limits, action_limits, dtype=np.float64)
        low_obs_limit = np.array([-1, -1, -1, -1, -1, -1, 0])
        high_obs_limit = np.array([1, 1, 1, 1, 1, 1, 1])
        self.observation_space = spaces.Box(low_obs_limit, high_obs_limit, dtype=np.float64)

        # VISUALIZATION
        self.viewer = None
        # all entities are composed of a Geom, a Transform (determining position) and a sprite
        self.agent_sprite = None
        self.agent_sprite_trans = None
        self.sheep_sprite = None
        self.sheep_sprite_trans = None
        self.lever_sprite = None
        self.lever_sprite_trans = None
        self.gate_sprite = None

        self.cage_sprite= None
        self.cage_sprite_trans = None

        self.stable_height = 0.5

        # background image is treated the same way
        self.background_sprite = None
        self.background_geom = None
        self.background_trans = None

        # Threshold for interaction
        self.lever_threshold= 0.2
        self.cage_threshold = 0.1

        # Scaling of action effect
        self.action_factor = 0.1

        # Range where the distance sensors detect walls
        self.distance_range = 0.1

        # Flag whether sheep is currently controlled
        self.gate_opened = False

        # Flag whether cage is currently carries
        self.cage_carried = True

        # Flaf whether sheep is currently hidden
        self.sheep_hidden = False

        self.sheep_in_cage = False

        self.sheep_velocity = 0.025

        self.last_action = np.array([-0.1, 0, 0], dtype=np.float64)
        self.wall_contacts = np.zeros(8, dtype=np.float64)
        self.t = 0
        self.t_render = 0

        self.roof_sprite = None
        self.reset_roof_sprite = True


    def seed(self, seed):
        self.r_seed = seed
        random.seed(seed)
        np.random.seed(seed)

    # ------------- STEP -------------
    def step(self, action):
        """
        Performs one step of the simulation
        :param action: next action to perform
        :return: next information, obtained reward, end of sequence?, additional inf
        """

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # Save the action
        self.last_action = action * self.action_factor

        # Move agent (and sheep) based on action
        self.agent_pos = self.agent_pos + self.last_action[:2]

        if self.cage_carried:
            self.cage_pos = self.cage_pos + self.last_action[:2]

        if not self.sheep_in_cage:
            if self.sheep_pos[1] >= 0.0 or self.gate_opened:
                self.sheep_pos[1] = self.sheep_pos[1] - self.sheep_velocity

            if self.sheep_pos[1] < self.stable_height and not self.gate_opened:
                self.sheep_hidden = True
                self.sheep_pos[1] = 0
            else:
                self.sheep_hidden = False
        else:
            if self.sheep_pos[1] > self.cage_pos[1]:
                self.sheep_pos[1] = self.cage_pos[1]

        # check boundaries of agent and patient position
        self._clip_positions()

        # Infrared sensor like measurement of wall distances
        self.wall_contacts = np.zeros(8, dtype=np.float64)
        double_pos_agent = np.append(self.agent_pos, self.agent_pos, 0)
        double_pos_sheep = np.append(self.sheep_pos, self.sheep_pos, 0)
        double_pos = np.append(double_pos_agent, double_pos_sheep, 0)
        self.wall_contacts = 1.0 - 10* abs(double_pos - self.position_limits)
        pos_diff = double_pos - self.position_limits
        self.wall_contacts[abs(pos_diff) > self.distance_range] = 0

        # Check if agent reached lever
        distance_agent_lever = np.linalg.norm(self.lever_pos - self.agent_pos)
        if distance_agent_lever < self.lever_threshold:
            self.gate_opened = True

        # Check if agent can carry cage
        distance_agent_cage = np.linalg.norm(self.cage_pos - self.agent_pos)

        if self.cage_carried and action[2] <= 0:
            self.cage_carried = False
        else:
            if not self.cage_carried and action[2] > 0 and distance_agent_cage < self.cage_threshold:
                self.cage_carried = True

        distance_sheep_cage = np.linalg.norm(self.cage_pos - self.sheep_pos)
        if not self.cage_carried and self.gate_opened and distance_sheep_cage < self.cage_threshold:
            self.sheep_in_cage = True


        # Check if sheep reached goal
        done = self.sheep_in_cage #or self.sheep_pos[1] < -1
        reward = 0.0
        if self.sheep_in_cage:
            reward = 1.0

        # Adjust sheep position such that it is in [-1, 1] for the observation
        norm_sheep_pos = np.copy(self.sheep_pos)
        norm_sheep_pos[0] -= 2.0

        if self.sheep_hidden:
            norm_sheep_pos[:] = 0.0

        # Create additional info vector
        gate_opened_array = np.array([0.0, 0.0, 0.0])
        if self.gate_opened:
            gate_opened_array[0] = 1.0
        if self.cage_carried:
            gate_opened_array[1] = 1.0
        if self.sheep_in_cage:
            gate_opened_array[2] = 1.0
        info = np.append(gate_opened_array, self.wall_contacts)

        norm_agent_pos = self.pos_to_obs(self.agent_pos)
        norm_cage_pos = self.pos_to_obs(self.cage_pos)
        agent_cage_pos = np.append(norm_agent_pos, norm_cage_pos, 0)

        observation = np.append(agent_cage_pos, norm_sheep_pos, 0)
        return np.append(observation, np.array([self.stable_height]), 0), reward, done, info

    def _clip_positions(self):
        np.clip(self.agent_pos, self.agent_pos_lower_limits, self.agent_pos_upper_limits, self.agent_pos)
        np.clip(self.cage_pos, self.agent_pos_lower_limits, self.agent_pos_upper_limits, self.cage_pos)
        if self.gate_opened:
            np.clip(self.sheep_pos, self.sheep_pos_lower_limits, self.sheep_pos_upper_limits, self.sheep_pos)
        else:
            np.clip(self.sheep_pos, self.sheep_pos_lower_limits_closed, self.sheep_pos_upper_limits, self.sheep_pos)


    # TRANSFORMATION OF FRAME OF REFERENCES

    def pos_to_obs(self, pos):
        """
        Normalizes positions as done when creating observation
        :param pos: position
        :return: position as observation
        """
        norm_pos = np.copy(pos)
        norm_pos[0] = (norm_pos[0] - 1) / 2
        norm_pos[1] = (norm_pos[1] + 0.5) / 2
        return norm_pos

    def obs_to_pos(self, obs):
        """
        Unnormalized observation to real underlying position
        :param pos: observation
        :return: observation as position
        """
        pos_from_obs = np.array([obs[0] * 2 + 1], obs[1] * 2 - 0.5)
        return pos_from_obs

    def obs_to_pos_batch(self, obs):
        """
        Normalizes positions as done when creating observation (batchwise)
        """
        assert len(obs.shape) == 3 and obs.shape[2] == 2
        real_pos = torch.clone(obs)
        real_pos[:, :, 0] = real_pos[:, :, 0] * 2 + 1
        real_pos[:, :, 1] = real_pos[:, :, 1] * 2 -0.5
        return real_pos

    def sheep_obs_to_pos_batch(self, sheep_obs):
        """
        Unnormalized observation to real underlying position (batchwise)
        """
        assert len(sheep_obs.shape) == 3 and sheep_obs.shape[2] == 2
        sheep_pos = torch.clone(sheep_obs)
        sheep_pos[:, :, 0] += 2
        return sheep_pos





    # ------------- RESET -------------


    def reset(self, reset_right = False, fixed_stable_height = False, generalize = False):
        """
        Randomly reset the simulation.
        :param reset_right: If agent is reset on the side of the scene
        :param fixed_stable_height: same fixed height of stable/wall
        :param generalize: generate wall heights outside of the normal range
        :return: first observation, additional info
        """

        self.agent_pos = (np.random.rand(2) - 0.5) * 2
        if reset_right:
            self.agent_pos[0] = self.agent_pos[0] + 2
        else:
            self.agent_pos[0] = (self.agent_pos[0] - 0.5) * 4 + 1
        self.agent_pos[1] = (self.agent_pos[1] - 1.0)



        self.cage_pos = self.agent_pos + (np.random.rand(2) - 0.5) * 0.1

        self.stable_height = random.random() * 0.3 + 0.35

        if fixed_stable_height:
            self.stable_height = 0.5
        elif generalize:
            factor = 1.0
            if random.random() < 0.5:
                factor = -1.0
            self.stable_height = factor * (random.random() * 0.15 + 0.15) + 0.5


        self.sheep_pos = (np.random.rand(2) - 0.5) + np.array([2.0, 0.0], dtype=np.float64)
        self.sheep_pos[1] = self.sheep_pos[1] * 0.01 + 0.95

        self.gate_opened = False

        self.sheep_velocity = 0.025

        self.cage_carried = True

        self.sheep_in_cage = False

        norm_sheep_pos = np.copy(self.sheep_pos)
        norm_sheep_pos[0] -= 2.0
        self.wall_contacts = np.zeros(8, dtype=np.float64)

        # Infrared sensor like measurement
        double_pos_agent = np.append(self.agent_pos, self.agent_pos, 0)
        double_pos_sheep = np.append(self.sheep_pos, self.sheep_pos, 0)
        double_pos = np.append(double_pos_agent, double_pos_sheep, 0)
        self.wall_contacts = 1.0 - 10 * abs(double_pos - self.position_limits)
        pos_diff = double_pos - self.position_limits
        self.wall_contacts[abs(pos_diff) > 0.1] = 0

        gate_opened_array = np.array([0.0, 0.0, 0.0])
        if self.gate_opened:
            gate_opened_array[0] = 1.0
        if self.cage_carried:
            gate_opened_array[1] = 1.0
        if self.sheep_in_cage:
            gate_opened_array[2] = 1.0
        info = np.append(gate_opened_array, self.wall_contacts)

        norm_agent_pos =  np.copy(self.agent_pos)
        norm_agent_pos[0] = (norm_agent_pos[0] -1)/2
        norm_agent_pos[1] = (norm_agent_pos[1] + 0.5)/2

        norm_cage_pos = np.copy(self.cage_pos)
        norm_cage_pos[0] = (norm_cage_pos[0] -1)/2
        norm_cage_pos[1] = (norm_cage_pos[1] + 0.5)/2

        agent_cage_pos = np.append(norm_agent_pos, norm_cage_pos, 0)

        observation = np.append(agent_cage_pos, norm_sheep_pos, 0)

        self.reset_roof_sprite = True

        return np.append(observation,  np.array([self.stable_height]), 0)

    # ------------- RENDERING -------------

    def _determine_sheep_sprite(self, action, t):
        """
        Finds the right sprite for the sheep depending on the last actions
        :param action: last action
        :param t: time
        :return: sprite number
        """
        return t % 2

    def _determine_agent_sprite(self, action, t):
        """
        Finds the right sprite for the agent depending on the last actions
        :param action: last action
        :param t: time
        :return: sprite number
        """
        if abs(action[1]) > abs(action[0]):
            # Left right dimension is stronger than up/down
            if action[1] < 0:
                # down:
                return 0 + t % 2
            else:
                # up
                return 2 + t % 2
        else:
            if action[0] < 0:
                # left:
                return 4 + t % 2
            else:
                # right
                return 6 + t % 2


    def _update_stable_roof_sprite(self, screen_height, sprite_scale):
        stable_roof_height = (self.stable_height - 0.25) / 0.5
        stable_roof_pixel_height = 40
        stable_roof_cropped_height = int(math.floor(stable_roof_height * stable_roof_pixel_height) + 5)
        stable_roof_image = pyglet.image.load(SCRIPT_PATH + "Shepherd_Sprites/stable_roof.png", decoder=PNGImageDecoder()).get_region(0, 0,
                                                                                                                   400,
                                                                                                                   stable_roof_cropped_height)
        stable_roof_pyglet_sprite = pyglet.sprite.Sprite(img=stable_roof_image)
        stable_roof_sprite_list = [stable_roof_pyglet_sprite]
        stable_roof_sprite = rendering.SpriteGeom(stable_roof_sprite_list)
        stable_roof_sprite_trans = rendering.Transform()
        stable_roof_sprite_trans.set_translation(0, screen_height * 3.03 / 4)
        stable_roof_sprite_trans.set_scale(sprite_scale, sprite_scale)
        stable_roof_sprite.add_attr(stable_roof_sprite_trans)
        stable_roof_sprite.set_z(3)
        self.viewer.add_geom(stable_roof_sprite)
        self.roof_sprite = stable_roof_sprite
        self.reset_roof_sprite = False

    def render(self, store_video=False, video_identifier=1, mode='human'):
        """
        Renders the simulation
        :param store_video: bool to save screenshots or not
        :param video_identifier: number to label video of this simulation
        :param mode: inherited from gym, currently not used
        """

        # Constant values of window size, sprite sizes, etc...
        screen_width = 1200  # pixels
        screen_height = 630  # pixels
        agent_sprite_width = 70  # pixels of sprite
        cage_sprite_width = 70
        sheep_sprite_width = 70
        lever_sprite_width = 70
        wall_pixel_width = 12
        scale = 300.0  # to compute from positions -> pixels
        foreground_sprite_scale = 2  # constant scaling of foreground sprites
        background_sprite_scale = 3  # constant scaling of walls, floor, etc...

        self.t += 1
        if self.t % 2 == 0:
            self.t_render += 1

        if self.viewer is None:
            # we create a new viewer (window)
            self.viewer = rendering.Viewer(screen_width, screen_height)

            glEnable(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

            # Agents sprite list [d, u, l, r] *[0, 1]
            agent_sprite_list = []
            sheep_sprite_list = []
            agent_sprite_names = ["d", "u", "l", "r"]
            for i in range(4):
                for j in range(2):
                    agent_sprite_file = SCRIPT_PATH + "RRC_Sprites/" + agent_sprite_names[i] + str(j + 1) + ".png"
                    agent_image = pyglet.image.load(agent_sprite_file, decoder=PNGImageDecoder())
                    agent_pyglet_sprite = pyglet.sprite.Sprite(img=agent_image)
                    agent_sprite_list.append(agent_pyglet_sprite)

            for s in range(2):
                sheep_sprite_file = SCRIPT_PATH + "Shepherd_Sprites/sheep" + str(s+1) + ".png"
                sheep_image = pyglet.image.load(sheep_sprite_file, decoder=PNGImageDecoder())
                sheep_pyglet_sprite = pyglet.sprite.Sprite(img=sheep_image)
                sheep_sprite_list.append(sheep_pyglet_sprite)

            self.agent_sprite = rendering.SpriteGeom(agent_sprite_list)
            self.agent_sprite_trans = rendering.Transform()
            self.agent_sprite.add_attr(self.agent_sprite_trans)
            self.viewer.add_geom(self.agent_sprite)

            self.sheep_sprite = rendering.SpriteGeom(sheep_sprite_list)
            self.sheep_sprite_trans = rendering.Transform()
            self.sheep_sprite.add_attr(self.sheep_sprite_trans)
            self.viewer.add_geom(self.sheep_sprite)

            lever_sprite_list = []
            for l in range(2):
                lever_image = pyglet.image.load(SCRIPT_PATH + "Shepherd_Sprites/lever" + str(l +1) + ".png", decoder=PNGImageDecoder())
                lever_pyglet_sprite = pyglet.sprite.Sprite(img=lever_image)
                lever_sprite_list.append(lever_pyglet_sprite)
            self.lever_sprite = rendering.SpriteGeom(lever_sprite_list)
            self.lever_sprite_trans = rendering.Transform()
            self.lever_sprite.add_attr(self.lever_sprite_trans)
            self.viewer.add_geom(self.lever_sprite)

            wall_image = pyglet.image.load(SCRIPT_PATH + "Shepherd_Sprites/long_wall.png", decoder=PNGImageDecoder())
            wall_pyglet_sprite = pyglet.sprite.Sprite(img=wall_image)
            wall_sprite_list = [wall_pyglet_sprite]
            wall_sprite = rendering.SpriteGeom(wall_sprite_list)
            wall_sprite_trans = rendering.Transform()
            wall_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            wall_sprite_trans.set_translation(screen_width - wall_pixel_width / 2.0 * background_sprite_scale, 0)
            wall_sprite.set_z(6)
            wall_sprite.add_attr(wall_sprite_trans)
            self.viewer.add_geom(wall_sprite)

            wall_2image = pyglet.image.load(SCRIPT_PATH + "Shepherd_Sprites/long_wall.png", decoder=PNGImageDecoder())
            wall_2pyglet_sprite = pyglet.sprite.Sprite(img=wall_2image)
            wall_2sprite_list = [wall_2pyglet_sprite]
            wall_2sprite = rendering.SpriteGeom(wall_2sprite_list)
            wall_2sprite_trans = rendering.Transform()
            wall_2sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            wall_2sprite_trans.set_translation(0 - wall_pixel_width / 2.0 * background_sprite_scale, 0)
            wall_2sprite.set_z(6)
            wall_2sprite.add_attr(wall_2sprite_trans)
            self.viewer.add_geom(wall_2sprite)


            cage_image = pyglet.image.load(SCRIPT_PATH + "Shepherd_Sprites/cage.png", decoder=PNGImageDecoder())
            cage_pyglet_sprite = pyglet.sprite.Sprite(img=cage_image)
            cage_sprite_list = [cage_pyglet_sprite]
            self.cage_sprite = rendering.SpriteGeom(cage_sprite_list)
            self.cage_sprite_trans = rendering.Transform()
            self.cage_sprite.add_attr(self.cage_sprite_trans)
            self.viewer.add_geom(self.cage_sprite)

            front_wall_image = pyglet.image.load(SCRIPT_PATH + "Shepherd_Sprites/brick_line.png", decoder=PNGImageDecoder())
            front_wall_pyglet_sprite = pyglet.sprite.Sprite(img=front_wall_image)
            front_wall_sprite_list = [front_wall_pyglet_sprite]
            front_wall_sprite = rendering.SpriteGeom(front_wall_sprite_list)
            front_wall_sprite_trans = rendering.Transform()
            front_wall_sprite_trans.set_translation(0, 0)
            front_wall_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            front_wall_sprite.add_attr(front_wall_sprite_trans)
            front_wall_sprite.set_z(6)
            self.viewer.add_geom(front_wall_sprite)

            stable_offset = -1

            stable_image = pyglet.image.load(SCRIPT_PATH + "Shepherd_Sprites/stable_walls.png", decoder=PNGImageDecoder())
            stable_pyglet_sprite = pyglet.sprite.Sprite(img=stable_image)
            stable_sprite_list = [stable_pyglet_sprite]
            stable_sprite = rendering.SpriteGeom(stable_sprite_list)
            stable_sprite_trans = rendering.Transform()
            stable_sprite_trans.set_translation(0, screen_height/2 + stable_offset)
            stable_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            stable_sprite.add_attr(stable_sprite_trans)
            self.viewer.add_geom(stable_sprite)

            stable_gate_sprite_list = []
            for g in range(2):
                stable_gate_image = pyglet.image.load(SCRIPT_PATH + "Shepherd_Sprites/gate" + str(g+1) + ".png", decoder=PNGImageDecoder())
                stable_gate_pyglet_sprite = pyglet.sprite.Sprite(img=stable_gate_image)
                stable_gate_sprite_list.append(stable_gate_pyglet_sprite)
            stable_gate_sprite = rendering.SpriteGeom(stable_gate_sprite_list)
            stable_gate_sprite_trans = rendering.Transform()
            stable_gate_sprite_trans.set_translation(screen_width/2, screen_height/2 + stable_offset)
            stable_gate_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            stable_gate_sprite.add_attr(stable_gate_sprite_trans)
            stable_gate_sprite.set_z(2)
            self.viewer.add_geom(stable_gate_sprite)
            self.gate_sprite = stable_gate_sprite

            stable_inners_image = pyglet.image.load(SCRIPT_PATH + "Shepherd_Sprites/gate_background.png", decoder=PNGImageDecoder())
            stable_inners_pyglet_sprite = pyglet.sprite.Sprite(img=stable_inners_image)
            stable_inners_sprite_list = [stable_inners_pyglet_sprite]
            stable_inners_sprite = rendering.SpriteGeom(stable_inners_sprite_list)
            stable_inners_sprite_trans = rendering.Transform()
            stable_inners_sprite_trans.set_translation(screen_width/2, screen_height/2 + stable_offset)
            stable_inners_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            stable_inners_sprite.add_attr(stable_inners_sprite_trans)
            stable_gate_sprite.set_z(1)
            self.viewer.add_geom(stable_inners_sprite)

            # Crop roof based on stable height
            self._update_stable_roof_sprite(screen_height=screen_height, sprite_scale=background_sprite_scale)

            background_image = pyglet.image.load(SCRIPT_PATH + "Shepherd_Sprites/grass_background.png", decoder=PNGImageDecoder())
            background_pyglet_sprite = pyglet.sprite.Sprite(img=background_image)
            background_sprite_list = [background_pyglet_sprite]
            background_sprite = rendering.SpriteGeom(background_sprite_list)
            background_sprite_trans = rendering.Transform()
            background_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            background_sprite.set_z(-2)
            background_sprite.add_attr(background_sprite_trans)
            self.viewer.add_geom(background_sprite)

        if self.reset_roof_sprite:
            assert self.roof_sprite is not None
            self.viewer.remove_geom(self.roof_sprite)
            self._update_stable_roof_sprite(screen_height=screen_height, sprite_scale=background_sprite_scale)



        # during video recording images of the simulation are saved
        if store_video:
            self.viewer.activate_video_mode("Video" + str(video_identifier) + "/")

        # determine the sprite position and size for
        # 1.  ... agent
        agent_x = (self.agent_pos[0] + 1) * scale
        agent_y = (self.agent_pos[1] + 1) * scale
        self.agent_sprite.set_z(5)
        agent_sprite_index = self._determine_agent_sprite(self.last_action, self.t_render)
        self.agent_sprite.alter_sprite_index(agent_sprite_index)
        self.agent_sprite_trans.set_scale(foreground_sprite_scale, foreground_sprite_scale)
        sprite_center = np.array([foreground_sprite_scale * agent_sprite_width / 2.0, 0.0])
        self.agent_sprite_trans.set_translation(agent_x - sprite_center[0], agent_y - sprite_center[1])

        # 2. ... the lever
        lever_x = (self.lever_pos[0] + 1) * scale
        lever_y = (self.lever_pos[1] + 1) * scale
        self.lever_sprite_trans.set_scale(foreground_sprite_scale, foreground_sprite_scale)
        lever_sprite_center = np.array([foreground_sprite_scale * lever_sprite_width / 2.0, foreground_sprite_scale * lever_sprite_width / 4.0])
        self.lever_sprite_trans.set_translation(lever_x - lever_sprite_center[0],
                                                   lever_y - lever_sprite_center[1])
        self.lever_sprite.set_z(4)
        if self.gate_opened:
            # Lever is switched
            self.lever_sprite.alter_sprite_index(1)
            self.gate_sprite.alter_sprite_index(1)

        else:
            self.lever_sprite.alter_sprite_index(0)
            self.gate_sprite.alter_sprite_index(0)
        if agent_y > lever_y + lever_sprite_width / 4.0:
            self.lever_sprite.set_z(6)

        # 3. ... the sheep
        sheep_x = (self.sheep_pos[0] + 1) * scale
        sheep_y = (self.sheep_pos[1] + 1) * scale
        self.sheep_sprite.set_z(-1)
        if self.gate_opened:
            self.sheep_sprite.set_z(3)
        self.sheep_sprite_trans.set_scale(foreground_sprite_scale, foreground_sprite_scale)
        sheep_sprite_center = np.array([foreground_sprite_scale * sheep_sprite_width / 2.0, 0.0])
        self.sheep_sprite_trans.set_translation(sheep_x - sheep_sprite_center[0], sheep_y - sheep_sprite_center[1])
        self.sheep_sprite.alter_sprite_index(self._determine_sheep_sprite(self.last_action, self.t_render))

        # 4. ... the cage
        cage_x = (self.cage_pos[0] + 1) * scale
        cage_y = (self.cage_pos[1] + 1) * scale
        cage_sprite_center = np.array([foreground_sprite_scale * cage_sprite_width / 2.0, 0.0])
        self.cage_sprite.set_z(6)
        if (not self.cage_carried and agent_y < cage_y) or (self.cage_carried and 2 <= agent_sprite_index < 4):
            # Cage is carried upwards or agent is above cage
            self.cage_sprite.set_z(4)
        self.cage_sprite_trans.set_scale(foreground_sprite_scale, foreground_sprite_scale)
        self.cage_sprite_trans.set_translation(cage_x - cage_sprite_center[0], cage_y - sprite_center[1])

        return self.viewer.render(mode == 'rgb_array')

    # ------------- CLOSE -------------
    def close(self):
        """
        Shut down the gym
        """
        if self.viewer:
            self.viewer.deactivate_video_mode()
            self.viewer.close()
            self.viewer = None
