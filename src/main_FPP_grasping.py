import argparse
import sys
import smart_settings
import os
import numpy as np
import torch
import datasets.fetch_grasp_dataset as fetch_grasp_dataset
import models.gatel0rd_model as GateL0RD
import models.rnn_model as RNNs
from utils.scheduled_sampling_helper import exponential_ss_prob
from utils.set_rs import set_rs

parser = argparse.ArgumentParser()
parser.add_argument("-config", help="Path to configuration file")
parser.add_argument("-seed", type=int, help="Seed for RNG")


if __name__ == '__main__':

    args = parser.parse_args(sys.argv[1:])
    params = smart_settings.load(args.config)

    # Random seed
    if args.seed is not None:
        rand_seed = args.seed
    else:
        rand_seed = params.rs
    set_rs(rand_seed)

    # Target directory
    target_dir = params.model_dir + str(rand_seed) + '/'
    os.makedirs(target_dir, exist_ok=True)

    # RNN type
    network_name = params.network_name

    # Dimensionality of the scenario
    action_dim = 4
    state_dim = 11
    input_dim = state_dim + action_dim
    output_dim = state_dim
    output_factor = 0.1

    if network_name == 'GateL0RD':
        net = GateL0RD.GateL0RDModel(input_dim=input_dim, output_dim=output_dim, latent_dim=params.latent_dim,
                                     feature_dim=params.feature_dim, num_layers=params.layer_num,
                                     num_layers_internal=params.num_layers_internal, reg_lambda=params.reg_lambda,
                                     f_pre_layers=params.preprocessing_layers,
                                     f_post_layers=params.postprocessing_layers,
                                     f_init_layers=params.warm_up_layers, f_init_inputs=params.warm_up_inputs)
    else:
        net = RNNs.RNNModel(input_dim=input_dim, output_dim=output_dim, latent_dim=params.latent_dim,
                               feature_dim=params.feature_dim, num_layers=params.layer_num,
                               rnn_type=network_name, f_pre_layers=params.preprocessing_layers,
                               f_post_layers=params.postprocessing_layers, f_init_layers=params.warm_up_layers,
                               f_init_inputs=params.warm_up_inputs)
    optimizer = net.get_optimizer(params.lr)


    # Load the data
    relevant_dims = np.arange(state_dim)
    dataset_split_rs = params.dataset_split_rs
    num_data_train = 3200
    num_data_test = 640
    num_data_test_gen = 3200
    batch_size = 128
    seq_len = params.seq_len

    train_dataloader, val_dataloader, test_dataloader, gen_dataloader = fetch_grasp_dataset.create_all_dataloaders(relevant_dims=relevant_dims,
                                                                                                                   seq_len=seq_len,
                                                                                                                   dataset_split_rs=dataset_split_rs,
                                                                                                                   num_data_train=num_data_train,
                                                                                                                   num_data_test=num_data_test,
                                                                                                                   num_data_gen=num_data_test_gen,
                                                                                                                   batch_size=batch_size)

    # Scheduled Sampling
    ss_slope = params.ss_slope
    start_steps = params.start_steps

    # Gradient norm clipping
    grad_clip = False
    if params.grad_clip_value > 0:
        grad_clip_value = params.grad_clip_value
        grad_clip = True

    t_validation = 100
    num_validations = 10 * params.train_len
    num_epochs = t_validation * num_validations

    val_MSE_over_t = np.zeros(num_validations, dtype='float64')
    test_MSE_over_t = np.zeros(num_validations, dtype='float64')
    gen_MSE_over_t = np.zeros(num_validations, dtype='float64')

    epoch_start = 0
    validations = 0

    for epoch in range(epoch_start, num_epochs):
        # Log one plot for every 1000th epoch
        if epoch%t_validation == 0:
            net = net.eval()


            # No scheduled sampling during validation
            val_ss = np.ones((seq_len, batch_size))
            val_ss[start_steps:seq_len, :] = -1

            with torch.no_grad():

                # Validation
                val_MSE_sum = 0.0
                val_count = 0

                for state, action, target in val_dataloader:
                    # Move sequence length to the front
                    states = state.permute(1, 0, 2).float()
                    actions = action.permute(1, 0, 2).float()
                    targets = target.permute(1, 0, 2).float()



                    y, z, gate_reg, _, _ = net.forward_n_step(obs_batch=states, train_schedule=val_ss,
                                                                     predict_deltas=True, factor_delta=output_factor,
                                                                     action_batch=actions)
                    real_next_states = states + targets
                    pred_next_states = y

                    MSE = net.MSE(real_next_states, pred_next_states).detach().item()
                    val_MSE_sum = val_MSE_sum + MSE
                    val_count = val_count + 1
                val_MSE_over_t[validations] = val_MSE_sum / val_count

                # Testing
                test_MSE_sum = 0.0
                test_count = 0
                for state, action, target in test_dataloader:

                    states = state.permute(1, 0, 2).float()
                    actions = action.permute(1, 0, 2).float()
                    targets = target.permute(1, 0, 2).float()

                    y, z, gate_reg, _, _ = net.forward_n_step(obs_batch=states, train_schedule=val_ss,
                                                                     predict_deltas=True, factor_delta=output_factor,
                                                                     action_batch=actions)
                    real_next_states = states + targets
                    pred_next_states = y

                    MSE = net.MSE(real_next_states, pred_next_states).detach().item()
                    test_MSE_sum = test_MSE_sum + MSE
                    test_count = test_count + 1
                test_MSE_over_t[validations] = test_MSE_sum / test_count



                gen_MSE_sum = 0.0
                gen_count = 0
                for state, action, target in gen_dataloader:
                    states = state.permute(1, 0, 2).float()
                    actions = action.permute(1, 0, 2).float()
                    targets = target.permute(1, 0, 2).float()

                    y, z, gate_reg, _, _ = net.forward_n_step(obs_batch=states, train_schedule=val_ss,
                                                                     predict_deltas=True,
                                                                     factor_delta=output_factor,
                                                                     action_batch=actions)
                    real_next_states = states + targets
                    pred_next_states = y

                    MSE = net.MSE(real_next_states, pred_next_states).detach().item()
                    gen_MSE_sum = gen_MSE_sum + MSE
                    gen_count = gen_count + 1
                gen_MSE_over_t[validations] = gen_MSE_sum / gen_count

            validations += 1

            dir_name_checkpoint = os.path.join(target_dir, "checkpoint_v" + str(validations))
            # Save a checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validations': validations
            }, dir_name_checkpoint)

        if validations < num_validations:

            net = net.train()

            # TRAINING:
            for state, action, target in train_dataloader:

                states = state.permute(1, 0, 2).float()
                actions = action.permute(1, 0, 2).float()
                targets = target.permute(1, 0, 2).float()

                # Scheduled sampling
                ss = np.ones((seq_len, batch_size))
                if ss_slope == 0:
                    ss_epsilon = 0.0
                else:
                    ss_epsilon = exponential_ss_prob(epoch=epoch, slope=ss_slope, min_value=0.05)
                    
                ss[start_steps:seq_len, :] = ss_epsilon

                optimizer.zero_grad()

                y, z, gate_reg, _, _ = net.forward_n_step(obs_batch=states, train_schedule=ss, predict_deltas=True,
                                                                 factor_delta=output_factor, action_batch=actions)

                real_next_states = states + targets
                pred_next_states = y

                loss = net.loss(y, real_next_states, gate_reg)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_value)
                optimizer.step()


    # Save val MSE as numpy file
    val_MSE_np_file = os.path.join(target_dir, "val_MSE_np")
    np.save(val_MSE_np_file, val_MSE_over_t)

    test_MSE_np_file = os.path.join(target_dir, "test_MSE_np")
    np.save(test_MSE_np_file, test_MSE_over_t)

    gen_MSE_np_file = os.path.join(target_dir, "gen_MSE_np")
    np.save(gen_MSE_np_file, gen_MSE_over_t)

    dir_name_checkpoint = os.path.join(target_dir, "final_checkpoint")
    # Save a checkpoint
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validations': validations
    }, dir_name_checkpoint)




