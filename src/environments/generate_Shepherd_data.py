import numpy as np
import shepherd_gym as gym
import random
import os

target_dir = "data/Shepherd/"
os.makedirs(target_dir, exist_ok=True)

num_data_per_dataset = 8000
seq_len = 101

random.seed(42)

# Sequences in which the gate is opened and agent starts on the left side
c = 0
memory_states = np.zeros((num_data_per_dataset, seq_len, 21))
for run in range(1000000, 2000000):
    env = gym.ShepherdGym(run)
    o_t = env.reset(reset_right=False, generalize=False)
    info = np.zeros(11)

    # Actions
    actions = (np.random.rand(3, seq_len) * 2 - 1.0)
    actions[2, :] *= 0.5
    factor = 1

    # States
    states = np.zeros((seq_len, 21), dtype=np.float64)
    gate_opened_t = -1
    for t in range(seq_len):

        if random.random() < 0.1:
            factor *= -1
        actions[2, t] += factor * 0.5

        oa_t = np.append(o_t, actions[:, t], 0)
        s_t = np.append(oa_t, info, 0)
        states[t, :] = s_t
        o_t, r_t, done, info = env.step(actions[:, t])
        if gate_opened_t == -1 and info[0] == 1:
            gate_opened_t = t

    if states[30, 10] == 0 and states[70, 10] == 1 and c < num_data_per_dataset:
        print(c, ":sheep freed at run", run, " with gate opened @ ", gate_opened_t)
        memory_states[c, :, :] = states[:, :]
        c += 1
    if c >= num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "shepherd_data_gateopen_startleft.npy"
np.save(filename, memory_states)
#torch.save(torch.from_numpy(memory_states).float(), filename)

# Sequences in which the gate is opened and agent starts on the right side
# Here left and up actions are sampled more frequently
c = 0
memory_states = np.zeros((num_data_per_dataset, seq_len, 21))
for run in range(2000000, 3000000):
    env = gym.ShepherdGym(run)
    o_t = env.reset(reset_right=True, generalize=False)

    # Actions
    actions = (np.random.rand(3, seq_len) * 2 - 1.0)
    actions[2, :] *= 0.5
    factor = 1

    # States
    states = np.zeros((seq_len, 21), dtype=np.float64)
    gate_opened_t = -1

    walk_up = False
    walk_left = False
    gate_opened = False

    for t in range(seq_len):

        if random.random() < 0.1:
            factor *= -1
        actions[2, t] += factor * 0.5

        if not walk_up:
            if random.random() < 0.02:
                walk_up = True
        else:
            if not gate_opened:
                actions[1, t] = actions[1, t] * 0.5 + 0.5

        if not walk_left:
            if random.random() < 0.02:
                walk_left = True
        else:
            if not gate_opened:
                actions[0, t] = actions[0, t] * 0.5 - 0.5
        oa_t = np.append(o_t, actions[:, t], 0)
        s_t = np.append(oa_t, info, 0)
        states[t, :] = s_t
        o_t, r_t, done, info = env.step(actions[:, t])

        if gate_opened_t == -1 and info[0] == 1:
            gate_opened_t = t
            gate_opened = True

    if states[30, 10] == 0 and states[70, 10] == 1 and states[100, 12] == 0 and c < num_data_per_dataset:
        print(c, ":sheep freed at run", run, " with gate opened @ ", gate_opened_t)
        memory_states[c, :, :] = states[:, :]
        c += 1
    if c >= num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "shepherd_data_gateopen_startright.npy"
np.save(filename, memory_states)
#torch.save(torch.from_numpy(memory_states).float(), filename)

# Sequences in which the gate is opened and agent starts on the right side and the sheep is caught
# Here left and up actions are sampled more frequently
c = 0
memory_states = np.zeros((num_data_per_dataset, seq_len, 21))
for run in range(3000000, 4000000):
    env = gym.ShepherdGym(run)
    o_t = env.reset(reset_right=True, generalize=False)

    # Actions
    actions = (np.random.rand(3, seq_len) * 2 - 1.0)
    actions[2, :] *= 0.5
    factor = 1

    # States
    states = np.zeros((seq_len, 21), dtype=np.float64)
    gate_opened_t = -1
    sheep_caught_t = -1

    walk_up = False
    walk_left = False
    gate_opened = False

    for t in range(seq_len):

        if random.random() < 0.1:
            factor *= -1
        actions[2, t] += factor * 0.5

        if not walk_up:
            if random.random() < 0.02:
                walk_up = True
        else:
            if not gate_opened:
                actions[1, t] = actions[1, t] * 0.5 + 0.5

        if not walk_left:
            if random.random() < 0.02:
                walk_left = True
        else:
            if not gate_opened:
                actions[0, t] = actions[0, t] * 0.5 - 0.5

        oa_t = np.append(o_t, actions[:, t], 0)
        s_t = np.append(oa_t, info, 0)
        states[t, :] = s_t
        o_t, r_t, done, info = env.step(actions[:, t])

        if gate_opened_t == -1 and info[0] == 1:
            gate_opened_t = t
            gate_opened = True

        if sheep_caught_t == -1 and info[2] == 1:
            sheep_caught_t = t

    if states[100, 12] == 1 and c < num_data_per_dataset:
        print(
        c, ":sheep caught at run ", run, " with gate opened @ ", gate_opened_t, " & sheep caught @ ", sheep_caught_t)
        memory_states[c, :, :] = states[:, :]
        c += 1
    if c >= num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "shepherd_data_sheepcaught_startright.npy"
np.save(filename, memory_states)
#torch.save(torch.from_numpy(memory_states).float(), filename)

# Sequences in which the gate remains closed
c = 0
memory_states = np.zeros((num_data_per_dataset, seq_len, 21))
for run in range(4000000, 5000000):
    env = gym.ShepherdGym(run)
    o_t = env.reset(reset_right=False, generalize=False)

    # Actions
    actions = (np.random.rand(3, seq_len) * 2 - 1.0)
    actions[2, :] *= 0.5
    factor = 1

    # States
    states = np.zeros((seq_len, 21), dtype=np.float64)
    gate_opened_t = -1
    for t in range(seq_len):

        if random.random() < 0.1:
            factor *= -1
        actions[2, t] += factor * 0.5

        oa_t = np.append(o_t, actions[:, t], 0)
        s_t = np.append(oa_t, info, 0)
        states[t, :] = s_t
        o_t, r_t, done, info = env.step(actions[:, t])
        if gate_opened_t == -1 and info[0] == 1:
            gate_opened_t = t

    if states[100, 10] == 0 and c < num_data_per_dataset:
        print(c, ":gate closed at run", run)
        memory_states[c, :, :] = states[:, :]
        c += 1
    if c >= num_data_per_dataset:
        env.close()
        break;
    env.close()
filename = target_dir + "shepherd_data_gateclosed_startleft.npy"
np.save(filename, memory_states)
#torch.save(torch.from_numpy(memory_states).float(), filename)