import numpy as np
import remote_control_gym as gym
import os

target_dir = "data/RRC/"
os.makedirs(target_dir, exist_ok=True)

num_data_per_dataset = 8000
seq_len = 50

# No control training data
rrc_states = np.zeros((num_data_per_dataset, seq_len, 15))
c = 0
for run in range(100000, 200000):
    env = gym.RemoteControlGym(run)
    t_mod = 0
    o_t, info = env.reset(with_info=True)
    actions = (np.random.rand(2, seq_len)*2 - 1.0)
    control = False
    states = np.zeros((seq_len, 15), dtype=np.float64)
    for t in range(seq_len):
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
    if c>=num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "GeneralizationNoControlDataTrain.npy"
np.save(filename, rrc_states)

# Control training data
rrc_states = np.zeros((num_data_per_dataset, seq_len, 15))
c = 0
for run in range(100000, 200000):
    env = gym.RemoteControlGym(run)
    t_mod = 0
    o_t, info = env.reset(with_info=True)
    actions = (np.random.rand(2, seq_len)*2 - 1.0)
    control = False
    states = np.zeros((seq_len, 15), dtype=np.float64)
    for t in range(seq_len):
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
    if c>=num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "GeneralizationControlDataTrain.npy"
np.save(filename, rrc_states)

# No control testing data
rrc_states = np.zeros((num_data_per_dataset, seq_len, 15))
c = 0
for run in range(0, 100000):
    env = gym.RemoteControlGym(run)
    t_mod = 0
    o_t, info = env.reset(with_info=True)
    actions = (np.random.rand(2, seq_len)*2 - 1.0)
    control = False
    states = np.zeros((seq_len, 15), dtype=np.float64)
    for t in range(seq_len):
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
    if c>=num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "GeneralizationNoControlDataTest.npy"
np.save(filename, rrc_states)

# Control testing data
rrc_states = np.zeros((num_data_per_dataset, seq_len, 15))
c = 0
for run in range(0, 100000):
    env = gym.RemoteControlGym(run)
    t_mod = 0
    o_t, info = env.reset(with_info=True)
    actions = (np.random.rand(2, seq_len)*2 - 1.0)
    control = False
    states = np.zeros((seq_len, 15), dtype=np.float64)
    for t in range(seq_len):
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
    if c>=num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "GeneralizationControlDataTest.npy"
np.save(filename, rrc_states)