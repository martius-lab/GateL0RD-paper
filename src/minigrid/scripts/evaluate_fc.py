import time
import torch
from torch_ac.utils.penv import ParallelEnv
import numpy as np
import utils



# Set seed for all randomness sources

def eval_fc(env_name, model, seed=0, procs = 16, episodes= 100, argmax = False, worst_episodes_to_show=10,
             memory=True, text= False, rnn_type = 'LSTM', memory_set_size=-1):


    utils.seed(seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load environments
    envs = []
    for i in range(procs):
        env = utils.make_env(env_name, seed + 10000 * i)
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agent

    model_dir = utils.get_model_dir(model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=argmax, num_envs=procs,
                    use_memory=memory, use_text=text, rnn_type=rnn_type, memory_set_size=memory_set_size)
    print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": [], "gating_per_episode": [], "successful_episode": []}

    # Run agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(procs, device=device)
    log_episode_num_frames = torch.zeros(procs, device=device)
    log_gating = torch.zeros(procs, device=device)

    while log_done_counter < episodes:
        actions, gates = agent.get_actions(obss)
        obss, rewards, dones, _ = env.step(actions)
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(procs, device=device)

        log_gating += torch.tensor(np.mean(gates, 1), device=device, dtype=torch.float)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                if log_episode_return[i].item() > 0:
                    logs["successful_episode"].append(1)
                else:
                    logs["successful_episode"].append(0)
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())
                gate_i = log_gating[i].item()/log_episode_num_frames[i].item()
                logs["gating_per_episode"].append(gate_i)

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask
        log_gating *= mask

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
    overall_gating = utils.synthesize(logs["gating_per_episode"])
    success_rate = utils.synthesize(logs["successful_episode"])

    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}" .format(num_frames, fps, duration, *return_per_episode.values(),*num_frames_per_episode.values()))

    # Print worst episodes

    n = worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))

    return return_per_episode['mean'], num_frames_per_episode['mean'], overall_gating['mean'], success_rate['mean']
