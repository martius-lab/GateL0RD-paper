import numpy as np
import remote_control_gym as gym
import os
import math

target_dir = "data/RRC2/"
os.makedirs(target_dir, exist_ok=True)

num_data_per_dataset = 8000
seq_len = 50

# No control of robot training data
rrc_states = np.zeros((num_data_per_dataset, seq_len, 15))
c = 0
for run in range(1900000, 2900000):
    env = gym.RemoteControlGym(run)
    t_mod = 0
    o_t, info = env.reset(with_info=True)
    angles = np.random.rand(1,seq_len) * 2 * math.pi
    random_x = np.sin(angles)
    random_y = np.cos(angles)
    random_xy =  np.concatenate((random_x, random_y), 0)
    factor = np.linspace(0.0001, 1, seq_len)
    actions = random_xy * factor
    control = False
    states = np.zeros((seq_len, 15), dtype=np.float64)
    for t in range(seq_len):
        #env.render()
        oa_t = np.append(o_t, actions[:, t_mod], 0)
        s_t = np.append(oa_t, info, 0)
        states[t, :] = s_t
        o_t, r_t, done, info = env.step(actions[:, t_mod])
        t_mod += 1
        if info[0] == 1.0:
            control = True
            break;
    if not control and c < num_data_per_dataset:
        rrc_states[c, :, :] = states[:, :]
        print(c, ":robot not controlled at run", run)
        c+= 1
    if c>= num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "BiasedNoControlDataTrain.npy"
np.save(filename, rrc_states)


# No control of robot testing data
rrc_states = np.zeros((num_data_per_dataset, seq_len, 15))
c = 0
for run in range(2900000, 3900000):
    env = gym.RemoteControlGym(run)
    t_mod = 0
    o_t, info = env.reset(with_info=True)
    angles = np.random.rand(1,seq_len) * 2 * math.pi
    random_x = np.sin(angles)
    random_y = np.cos(angles)
    random_xy =  np.concatenate((random_x, random_y), 0)
    factor = np.linspace(0.0001, 1, seq_len)
    actions = random_xy * factor
    control = False
    states = np.zeros((seq_len, 15), dtype=np.float64)
    for t in range(seq_len):
        #env.render()
        oa_t = np.append(o_t, actions[:, t_mod], 0)
        s_t = np.append(oa_t, info, 0)
        states[t, :] = s_t
        o_t, r_t, done, info = env.step(actions[:, t_mod])
        t_mod += 1
        if info[0] == 1.0:
            control = True
            break;
    if not control and c < num_data_per_dataset:
        rrc_states[c, :, :] = states[:, :]
        print(c, ":robot not controlled at run", run)
        c+= 1
    if c>= num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "BiasedNoControlDataTest.npy"
np.save(filename, rrc_states)

# Control of robot training data
rrc_states = np.zeros((num_data_per_dataset, seq_len, 15))
c = 0
for run in range(3900000, 4900000):
    env = gym.RemoteControlGym(run)
    t_mod = 0
    o_t, info = env.reset(with_info=True)
    angles = np.random.rand(1,seq_len) * 2 * math.pi
    random_x = np.sin(angles)
    random_y = np.cos(angles)
    random_xy =  np.concatenate((random_x, random_y), 0)
    factor = np.linspace(0.0001, 1, seq_len)
    actions = random_xy * factor
    control = False
    states = np.zeros((seq_len, 15), dtype=np.float64)
    for t in range(seq_len):
        #env.render()
        oa_t = np.append(o_t, actions[:, t_mod], 0)
        s_t = np.append(oa_t, info, 0)
        states[t, :] = s_t
        o_t, r_t, done, info = env.step(actions[:, t_mod])
        t_mod += 1
        if info[0] == 1.0:
            control = True
    if control and c < num_data_per_dataset:
        rrc_states[c, :, :] = states[:, :]
        print(c, ":robot controlled at run", run)
        c+= 1
    if c>= num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "BiasedControlDataTrain.npy"
np.save(filename, rrc_states)


# Control of robot testing data
rrc_states = np.zeros((num_data_per_dataset, seq_len, 15))
c = 0
for run in range(4900000, 5900000):
    env = gym.RemoteControlGym(run)
    t_mod = 0
    o_t, info = env.reset(with_info=True)
    angles = np.random.rand(1, seq_len) * 2 * math.pi
    random_x = np.sin(angles)
    random_y = np.cos(angles)
    random_xy = np.concatenate((random_x, random_y), 0)
    factor = np.linspace(0.0001, 1, seq_len)
    actions = random_xy * factor
    control = False
    states = np.zeros((seq_len, 15), dtype=np.float64)
    for t in range(seq_len):
        # env.render()
        oa_t = np.append(o_t, actions[:, t_mod], 0)
        s_t = np.append(oa_t, info, 0)
        states[t, :] = s_t
        o_t, r_t, done, info = env.step(actions[:, t_mod])
        t_mod += 1
        if info[0] == 1.0:
            control = True
    if control and c < num_data_per_dataset:
        rrc_states[c, :, :] = states[:, :]
        print(c, ":robot controlled at run", run)
        c += 1
    if c >= num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "BiasedControlDataTest.npy"
np.save(filename, rrc_states)