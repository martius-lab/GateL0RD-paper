"""
Simple control task where an agent can control a robot after accessing a computer.
The goal is to bring the robot to a goal location

The scenario is implemented in the OpenAi Gym interface (gym.openai.com)

"""

import gym
from gym import spaces, logger
import remote_control_gym_rendering as rendering
import numpy as np
import pyglet
import random
from pyglet.gl import *
from pyglet.image.codecs.png import PNGImageDecoder

import os

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'

class RemoteControlGym(gym.Env):
    """
    Simple gym task for controlling a boy and a robot in a 2D environment

    Observation (4 dimensional):
    - Position of agent (x, y)
    - Position of robot (x, y)

    Action (2-dimensional):
    - Movement of agent (x, y)

    Additional information (may be used as additional observation, 9-dimensional)
    - Is the robot controlled?
    - Distance to wall for agent (N, E, S, W)
    - Distance to wall for agent (N, E, S, W)

    Reward is obtained when the robot is moved to the goal position.
    The robot can be moved when the agent reaches a computer.
    Agent and robot starting positions are randomly sampled for each simulation.
    Goal and computer position do not change over simulations.

    """

    def __init__(self, r_seed=42):
        """
        :param r_seed: random seed
        """
        super().__init__()
        np.random.seed(r_seed)
        self.agent_pos = (np.random.rand(2) - 0.5) * 2
        self.robot_pos = (np.random.rand(2) - 0.5) * 2 + np.array([2.0, 0.0], dtype=np.float64)
        self.computer_pos = np.array([0.5, 0.5], dtype=np.float64)
        self.goal_pos = np.array([2.5, -0.5], dtype=np.float64)
        self.r_seed = r_seed

        self.agent_pos_upper_limits = np.array([0.95, 0.95], dtype=np.float64)
        self.agent_pos_lower_limits = np.array([-0.95, -0.95], dtype=np.float64)
        self.robot_pos_upper_limits = np.array([2.95, 0.95], dtype=np.float64)
        self.robot_pos_lower_limits = np.array([1.05, -0.95], dtype=np.float64)
        self.position_limits = np.array([0.95, 0.95, -0.95, -0.95, 2.95, 0.95, 1.05, -0.95], dtype=np.float64)
        action_limits = np.array([1, 1])
        obs_limits = np.array([0.95, 0.95, 0.95, 0.95])
        self.action_space = spaces.Box(-1 * action_limits, action_limits, dtype=np.float64)
        self.observation_space = spaces.Box(-1 * obs_limits, obs_limits, dtype=np.float64)


        # VISUALIZATION
        self.viewer = None
        # all entities are composed of a Geom, a Transform (determining position) and a sprite
        self.agent_sprite = None
        self.agent_sprite_trans = None
        self.robot_sprite = None
        self.robot_sprite_trans = None
        self.computer_sprite = None
        self.computer_sprite_trans = None
        self.goal_sprite = None
        self.goal_sprite_trans = None

        # background image is treated the same way
        self.background_sprite = None
        self.background_geom = None
        self.background_trans = None

        # Threshold for reaching the goal
        self.goal_threshold = 0.1

        # Scaling of action effect
        self.action_factor = 0.1

        # Range where the distance sensors detect walls
        self.distance_range = 0.1

        # Flag whether robot is currently controlled
        self.robot_controlled = False

        self.last_action = np.array([-0.1, 0], dtype=np.float64)
        self.wall_contacts = np.zeros(8, dtype=np.float64)
        self.t = 0
        self.t_render = 0


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

        # Move agent (and robot) based on action
        self.agent_pos = self.agent_pos + self.last_action
        if self.robot_controlled:
            self.robot_pos = self.robot_pos + self.last_action

        # check boundaries of agent and patient position
        self._clip_positions()

        # Infrared sensor like measurement of wall distances
        self.wall_contacts = np.zeros(8, dtype=np.float64)
        double_pos_agent = np.append(self.agent_pos, self.agent_pos, 0)
        double_pos_robot = np.append(self.robot_pos, self.robot_pos, 0)
        double_pos = np.append(double_pos_agent, double_pos_robot, 0)
        self.wall_contacts = 1.0 - 10* abs(double_pos - self.position_limits)
        pos_diff = double_pos - self.position_limits
        self.wall_contacts[abs(pos_diff) > self.distance_range] = 0

        # Check if agent reached computer
        distance_agent_computer = np.linalg.norm(self.computer_pos - self.agent_pos)
        if distance_agent_computer < self.goal_threshold:
            self.robot_controlled = True

        # Check if robot reached goal
        distance_robot_goal = np.linalg.norm(self.goal_pos - self.robot_pos)
        done = distance_robot_goal < self.goal_threshold
        reward = 0.0
        if done:
            reward = 1.0

        # Adjust robot position such that it is in [-1, 1] for the observation
        norm_robot_pos = np.copy(self.robot_pos)
        norm_robot_pos[0] -= 2.0

        # Create additional info vector
        robot_controlled_array = np.array([0.0])
        if self.robot_controlled:
            robot_controlled_array[0] = 1.0
        info = np.append(robot_controlled_array, self.wall_contacts)

        observation = np.append(self.agent_pos.flat, norm_robot_pos.flat, 0)
        return observation, reward, done, info

    def _clip_positions(self):
        np.clip(self.agent_pos, self.agent_pos_lower_limits, self.agent_pos_upper_limits, self.agent_pos)
        np.clip(self.robot_pos, self.robot_pos_lower_limits, self.robot_pos_upper_limits, self.robot_pos)

    # ------------- RESET -------------
    def reset(self, with_info=False):
        """
        Randomly reset the simulation.
        :param with_info: include additional info (robot control, wall sensors) in output
        :return: first observation, additional info
        """

        self.agent_pos = (np.random.rand(2) - 0.5) * 2
        self.robot_pos = (np.random.rand(2) - 0.5) * 2 + np.array([2.0, 0.0], dtype=np.float64)
        self.computer_pos = np.array([0.5, 0.5], dtype=np.float64)
        self.goal_pos = np.array([2.5, -0.5], dtype=np.float64)
        self.robot_controlled = False

        norm_robot_pos = np.copy(self.robot_pos)
        norm_robot_pos[0] -= 2.0
        self.wall_contacts = np.zeros(8, dtype=np.float64)
        # Infrared sensor like measurement
        double_pos_agent = np.append(self.agent_pos, self.agent_pos, 0)
        double_pos_robot = np.append(self.robot_pos, self.robot_pos, 0)
        double_pos = np.append(double_pos_agent, double_pos_robot, 0)
        self.wall_contacts = 1.0 - 10 * abs(double_pos - self.position_limits)
        pos_diff = double_pos - self.position_limits
        self.wall_contacts[abs(pos_diff) > 0.1] = 0

        o_init = np.append(self.agent_pos.flat, norm_robot_pos.flat, 0)

        if with_info:
            # Create additional info vector
            robot_controlled_array = np.array([0.0])
            if self.robot_controlled:
                robot_controlled_array[0] = 1.0
            info_init = np.append(robot_controlled_array, self.wall_contacts)
            return o_init, info_init

        return o_init

    # ------------- RENDERING -------------

    def _determine_robot_sprite(self, action, t):
        """
        Finds the right sprite for the robot depending on the last actions
        :param action: last action
        :param t: time
        :return: sprite number
        """
        if self.robot_controlled:
            return t % 4

        return 0

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
                #up
                return 2 + t % 2
        else:
            if action[0] < 0:
                # left:
                return 4 + t % 2
            else:
                #right
                return 6 + t % 2

    def render(self, store_video=False, video_identifier=1,  mode='human'):
        """
        Renders the simulation
        :param store_video: bool to save screenshots or not
        :param video_identifier: number to label video of this simulation
        :param mode: inherited from gym, currently not used
        """
        
        # Constant values of window size, sprite sizes, etc... 
        screen_width = 1200  #pixels
        screen_height = 630  #pixels
        agent_sprite_width = 70 # pixels of sprite
        robot_sprite_width = 70
        computer_sprite_width = 70
        wall_pixel_width = 12
        goal_sprite_width = 16
        goal_sprite_height = 16
        scale = 300.0  # to compute from positions -> pixels
        foreground_sprite_scale = 2 # constant scaling of foreground sprites
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
            robot_sprite_list = []
            agent_sprite_names = ["d", "u", "l", "r"]
            for i in range(4):
                robot_sprite_file = SCRIPT_PATH + "RRC_Sprites/rob" + str(i+1) + ".png"
                robot_image = pyglet.image.load(robot_sprite_file, decoder=PNGImageDecoder())
                robot_pyglet_sprite = pyglet.sprite.Sprite(img=robot_image)
                robot_sprite_list.append(robot_pyglet_sprite)
                for j in range(2):
                    agent_sprite_file = SCRIPT_PATH + "RRC_Sprites/" + agent_sprite_names[i] + str(j+1) + ".png"
                    agent_image = pyglet.image.load(agent_sprite_file, decoder=PNGImageDecoder())
                    agent_pyglet_sprite = pyglet.sprite.Sprite(img=agent_image)
                    agent_sprite_list.append(agent_pyglet_sprite)

            self.agent_sprite = rendering.SpriteGeom(agent_sprite_list)
            self.agent_sprite_trans = rendering.Transform()
            self.agent_sprite.add_attr(self.agent_sprite_trans)
            self.viewer.add_geom(self.agent_sprite)

            self.robot_sprite = rendering.SpriteGeom(robot_sprite_list)
            self.robot_sprite_trans = rendering.Transform()
            self.robot_sprite.add_attr(self.robot_sprite_trans)
            self.viewer.add_geom(self.robot_sprite)

            goal_image = pyglet.image.load(SCRIPT_PATH + "RRC_Sprites/target.png", decoder=PNGImageDecoder())
            goal_pyglet_sprite =  pyglet.sprite.Sprite(img=goal_image)
            goal_sprite_list = [goal_pyglet_sprite]
            self.goal_sprite = rendering.SpriteGeom(goal_sprite_list)
            self.goal_sprite_trans = rendering.Transform()
            self.goal_sprite.add_attr(self.goal_sprite_trans)
            self.viewer.add_geom(self.goal_sprite)

            computer_image = pyglet.image.load(SCRIPT_PATH + "RRC_Sprites/terminal2.png", decoder=PNGImageDecoder())
            computer_pyglet_sprite = pyglet.sprite.Sprite(img=computer_image)
            computer_sprite_list = [computer_pyglet_sprite]
            self.computer_sprite = rendering.SpriteGeom(computer_sprite_list)
            self.computer_sprite_trans = rendering.Transform()
            self.computer_sprite.add_attr(self.computer_sprite_trans)
            self.viewer.add_geom(self.computer_sprite)

            wall_image = pyglet.image.load(SCRIPT_PATH + "RRC_Sprites/long_wall.png", decoder=PNGImageDecoder())
            wall_pyglet_sprite = pyglet.sprite.Sprite(img=wall_image)
            wall_sprite_list = [wall_pyglet_sprite]
            wall_sprite = rendering.SpriteGeom(wall_sprite_list)
            wall_sprite_trans = rendering.Transform()
            wall_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            wall_sprite_trans.set_translation(screen_width/2.0 - wall_pixel_width/2.0 * background_sprite_scale, 0)
            wall_sprite.set_z(3)
            wall_sprite.add_attr(wall_sprite_trans)
            self.viewer.add_geom(wall_sprite)

            wall_2image = pyglet.image.load(SCRIPT_PATH + "RRC_Sprites/long_wall.png", decoder=PNGImageDecoder())
            wall_2pyglet_sprite = pyglet.sprite.Sprite(img=wall_2image)
            wall_2sprite_list = [wall_2pyglet_sprite]
            wall_2sprite = rendering.SpriteGeom(wall_2sprite_list)
            wall_2sprite_trans = rendering.Transform()
            wall_2sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            wall_2sprite_trans.set_translation(0 - wall_pixel_width / 2.0 * background_sprite_scale, 0)
            wall_2sprite.set_z(3)
            wall_2sprite.add_attr(wall_2sprite_trans)
            self.viewer.add_geom(wall_2sprite)

            wall_3image = pyglet.image.load(SCRIPT_PATH + "RRC_Sprites/long_wall.png", decoder=PNGImageDecoder()) # "Sprites/wall_longest3.png"
            wall_3pyglet_sprite = pyglet.sprite.Sprite(img=wall_3image)
            wall_3sprite_list = [wall_3pyglet_sprite]
            wall_3sprite = rendering.SpriteGeom(wall_3sprite_list)
            wall_3sprite_trans = rendering.Transform()
            wall_3sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            wall_3sprite_trans.set_translation(screen_width - wall_pixel_width / 2.0 * background_sprite_scale, 0)
            wall_3sprite.set_z(3)
            wall_3sprite.add_attr(wall_3sprite_trans)
            self.viewer.add_geom(wall_3sprite)

            back_wall_image = pyglet.image.load(SCRIPT_PATH + "RRC_Sprites/back_wall.png", decoder=PNGImageDecoder())
            back_wall_pyglet_sprite = pyglet.sprite.Sprite(img=back_wall_image)
            back_wall_sprite_list = [back_wall_pyglet_sprite]
            back_wall_sprite = rendering.SpriteGeom(back_wall_sprite_list)
            back_wall_sprite_trans = rendering.Transform()
            back_wall_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            back_wall_pixel_height = 20
            back_wall_sprite_trans.set_translation(0, screen_height - back_wall_pixel_height)
            back_wall_sprite.add_attr(back_wall_sprite_trans)
            self.viewer.add_geom(back_wall_sprite)

            front_wall_image = pyglet.image.load(SCRIPT_PATH + "RRC_Sprites/grey_line.png", decoder=PNGImageDecoder())
            front_wall_pyglet_sprite = pyglet.sprite.Sprite(img=front_wall_image)
            front_wall_sprite_list = [front_wall_pyglet_sprite]
            front_wall_sprite = rendering.SpriteGeom(front_wall_sprite_list)
            front_wall_sprite_trans = rendering.Transform()
            front_wall_sprite_trans.set_translation(0, 0)
            front_wall_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            front_wall_sprite.add_attr(front_wall_sprite_trans)
            self.viewer.add_geom(front_wall_sprite)

            background_image = pyglet.image.load(SCRIPT_PATH + "RRC_Sprites/grey_wood_background.png", decoder=PNGImageDecoder())
            background_pyglet_sprite = pyglet.sprite.Sprite(img=background_image)
            background_sprite_list = [background_pyglet_sprite]
            background_sprite = rendering.SpriteGeom(background_sprite_list)
            background_sprite_trans = rendering.Transform()
            background_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
            background_sprite.set_z(-1)
            background_sprite.add_attr(background_sprite_trans)
            self.viewer.add_geom(background_sprite)

        # during video recording images of the simulation are saved
        if store_video:
            self.viewer.activate_video_mode("Video" + str(video_identifier) + "/")

        # determine the sprite position and size for
        # 1.  ... agent
        agent_x = (self.agent_pos[0] + 1) * scale
        agent_y = (self.agent_pos[1] + 1) * scale
        self.agent_sprite.set_z(1)
        self.agent_sprite.alter_sprite_index(self._determine_agent_sprite(self.last_action, self.t_render))
        self.agent_sprite_trans.set_scale(foreground_sprite_scale, foreground_sprite_scale)
        sprite_center = np.array([foreground_sprite_scale * agent_sprite_width / 2.0, 0.0])
        self.agent_sprite_trans.set_translation(agent_x - sprite_center[0], agent_y - sprite_center[1])

        # 2. ... the computer
        computer_x = (self.computer_pos[0] + 1) * scale
        computer_y = (self.computer_pos[1] + 1) * scale
        self.computer_sprite_trans.set_scale(foreground_sprite_scale, foreground_sprite_scale)
        computer_sprite_center = np.array([foreground_sprite_scale * computer_sprite_width / 2.0, 0])
        self.computer_sprite_trans.set_translation(computer_x - computer_sprite_center[0], computer_y - computer_sprite_center[1])
        self.computer_sprite.set_z(0)
        if agent_y > computer_y + computer_sprite_width / 4.0:
            self.computer_sprite.set_z(2)

        # 3. ... the goal
        goal_x = (self.goal_pos[0] + 1) * scale
        goal_y = (self.goal_pos[1] + 1) * scale
        self.goal_sprite.set_z(0)
        self.goal_sprite_trans.set_scale(background_sprite_scale, background_sprite_scale)
        goal_sprite_center = np.array([background_sprite_scale * goal_sprite_width / 2.0, background_sprite_scale * goal_sprite_height / 2.0])
        self.goal_sprite_trans.set_translation(goal_x - goal_sprite_center[0], goal_y - goal_sprite_center[1])

        # 4. ... the robot
        robot_x = (self.robot_pos[0] + 1) * scale
        robot_y = (self.robot_pos[1] + 1) * scale
        self.robot_sprite.set_z(1)
        self.robot_sprite_trans.set_scale(foreground_sprite_scale, foreground_sprite_scale)
        robot_sprite_center = np.array([foreground_sprite_scale * robot_sprite_width / 2.0, 0.0])
        self.robot_sprite_trans.set_translation(robot_x - robot_sprite_center[0], robot_y - robot_sprite_center[1])
        self.robot_sprite.alter_sprite_index(self._determine_robot_sprite(self.last_action, self.t_render))

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
