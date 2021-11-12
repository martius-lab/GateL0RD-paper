import time
import datetime
import torch
import torch_ac
import tensorboardX

import utils
from model import ACModel


def train_fc(algo, env, model, seed=1, log_interval=1, save_interval=10, procs = 16, frames = 10**7, epochs=4,
             batch_size=256, frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95, entropy_coef=0.01,
             value_loss_coef=0.5, max_grad_norm=0.5, optim_eps=1e-8, optim_alpha=0.99, clip_eps=0.2, recurrence=1,
             rnn_type = 'LSTM', reg_lambda = 0.0, text=False, memory_set_size=-1):
    mem = recurrence > 1

    #reloaded = False

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{env}_{algo}_seed{seed}_{date}"

    model_name = model or default_model_name
    model_dir = utils.get_model_dir(model_name)
    #assert False, model_dir

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Set seed for all randomness sources

    utils.seed(seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(procs):
        envs.append(utils.make_env(env, seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model

    acmodel = ACModel(device, obs_space, envs[0].action_space, mem, text, rnn_type, memory_set_size)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
        #reloaded = True
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, frames_per_proc, discount, lr, gae_lambda,
                            entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                            optim_alpha, optim_eps, preprocess_obss)
    elif algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, frames_per_proc, discount, lr, gae_lambda,
                            entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                            optim_eps, clip_eps, epochs, batch_size, preprocess_obss, reg_lambda=reg_lambda)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1



        # Print logs

        if update % log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status
        #if save_interval > 0 and update % save_interval == 0:
    status = {"num_frames": num_frames, "update": update, "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
    if hasattr(preprocess_obss, "vocab"):
        status["vocab"] = preprocess_obss.vocab.vocab
    utils.save_status(status, model_dir)
    txt_logger.info("Status saved")

