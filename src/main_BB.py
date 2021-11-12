import argparse
import sys
import smart_settings
import os
import numpy as np
import torch
import models.gatel0rd_model as GateL0RD
import models.rnn_model as RNNs
from utils.scheduled_sampling_helper import exponential_ss_prob
from utils.set_rs import set_rs
from utils.gating_analysis import calc_gate_rate,calc_gate_dims_used,calc_gate_open_times
import datasets.billiard_ball_dataloader as bb_dataloader


parser = argparse.ArgumentParser()
parser.add_argument("-config", help="Path to configuration file")
parser.add_argument("-seed", type=int, help="Seed for RNG")


if __name__ == '__main__':

    args = parser.parse_args(sys.argv[1:])
    params = smart_settings.load(args.config)

    # RNN type
    network_name = params.network_name

    # Random seed
    if args.seed is not None:
        rand_seed = args.seed
    else:
        rand_seed = params.rs
    set_rs(rand_seed)

    # Target directory:
    target_dir = params.model_dir + str(rand_seed) + '/'
    os.makedirs(target_dir, exist_ok=True)


    # General dimensionality of the scenario
    input_dim = 2
    output_dim = 2
    factor_output = 0.1

    if network_name == 'GateL0RD':
        net = GateL0RD.GateL0RDModel(input_dim=input_dim, output_dim=output_dim, latent_dim=params.latent_dim,
                                     feature_dim=params.feature_dim, num_layers=params.layer_num,
                                     num_layers_internal=params.num_layers_internal, reg_lambda=params.reg_lambda,
                                     f_pre_layers=params.preprocessing_layers,
                                     f_post_layers=params.postprocessing_layers,
                                     f_init_layers=params.warm_up_layers, f_init_inputs=params.warm_up_inputs,
                                     stochastic_gates=params.stochastic_gates,
                                     gate_noise_level=params.gate_noise_level, gate_type=params.gate_type)
    else:
        net = RNNs.RNNModel(input_dim=input_dim, output_dim=output_dim, latent_dim=params.latent_dim,
                               feature_dim=params.feature_dim, num_layers=params.layer_num,
                               rnn_type=network_name, f_pre_layers=params.preprocessing_layers,
                               f_post_layers=params.postprocessing_layers, f_init_layers=params.warm_up_layers,
                               f_init_inputs=params.warm_up_inputs)

    optimizer = net.get_optimizer(params.lr)


    # Load the datasets
    dataset_split_rs = params.dataset_split_rs
    seq_len = 50
    num_subdatasets = 16
    subdataset_batch_size = 8
    batch_size = subdataset_batch_size * num_subdatasets

    train_dataloaders, val_dataloaders, test_dataloaders = bb_dataloader.get_billiard_ball_dataloaders(dataset_split_rs=dataset_split_rs,
                                                                                                       seq_len=seq_len,
                                                                                                       subdataset_batch_size=subdataset_batch_size)

    # Scheduled Sampling
    ss_slope = float(params.ss_slope)
    start_steps = params.start_steps
    teacherforcing = params.teacherforcing

    # Gradient norm clipping
    grad_clip = False
    grad_clip_value = 1000.0
    if params.grad_clip_value > 0:
        grad_clip_value = params.grad_clip_value
        grad_clip = True

    t_validation = 100
    num_validations = 10 * params.train_len
    num_epochs = t_validation * num_validations


    # Keep track of prediction errors
    val_MSE_over_t = np.zeros(num_validations, dtype='float64')
    test_MSE_over_t = np.zeros(num_validations, dtype='float64')
    test_gate_rate = np.zeros(num_validations, dtype='float64')
    test_gate_dims = np.zeros(num_validations, dtype='float64')
    test_gate_times = np.zeros(num_validations, dtype='float64')

    epoch_start= 0
    validations = 0

    for epoch in range(epoch_start, num_epochs):

        # Log one plot for every 1000th epoch
        if epoch%t_validation == 0:

            net = net.eval()

            # No scheduled sampling during validation
            val_ss = np.ones((seq_len, batch_size))
            val_ss[start_steps:seq_len, :] = -1

            with torch.no_grad():


                # Validation data
                val_MSE_sum = 0.0
                val_count = 0
                for data_a, data_b, data_c, data_d, data_e, data_f, data_g, data_h, data_i, data_j, data_k, data_l, \
                    data_m, data_n, data_o, data_p in zip(*val_dataloaders):

                    input_a = data_a[0].permute(1, 0, 2)
                    target_a = data_a[1].permute(1, 0, 2)
                    input_b = data_b[0].permute(1, 0, 2)
                    target_b = data_b[1].permute(1, 0, 2)
                    input_c = data_c[0].permute(1, 0, 2)
                    target_c = data_c[1].permute(1, 0, 2)
                    input_d = data_d[0].permute(1, 0, 2)
                    target_d = data_d[1].permute(1, 0, 2)
                    input_e = data_e[0].permute(1, 0, 2)
                    target_e = data_e[1].permute(1, 0, 2)
                    input_f = data_f[0].permute(1, 0, 2)
                    target_f = data_f[1].permute(1, 0, 2)

                    input_g = data_g[0].permute(1, 0, 2)
                    target_g = data_g[1].permute(1, 0, 2)
                    input_h = data_h[0].permute(1, 0, 2)
                    target_h = data_h[1].permute(1, 0, 2)
                    input_i = data_i[0].permute(1, 0, 2)
                    target_i = data_i[1].permute(1, 0, 2)
                    input_j = data_j[0].permute(1, 0, 2)
                    target_j = data_j[1].permute(1, 0, 2)
                    input_k = data_k[0].permute(1, 0, 2)
                    target_k = data_k[1].permute(1, 0, 2)

                    input_l = data_l[0].permute(1, 0, 2)
                    target_l = data_l[1].permute(1, 0, 2)
                    input_m = data_m[0].permute(1, 0, 2)
                    target_m = data_m[1].permute(1, 0, 2)
                    input_n = data_n[0].permute(1, 0, 2)
                    target_n = data_n[1].permute(1, 0, 2)
                    input_o = data_o[0].permute(1, 0, 2)
                    target_o = data_o[1].permute(1, 0, 2)

                    input_p = data_p[0].permute(1, 0, 2)
                    target_p = data_p[1].permute(1, 0, 2)


                    inp = torch.cat((input_a, input_b, input_c, input_d, input_e, input_f, input_g, input_h, input_i,
                                     input_j, input_k, input_l, input_m, input_n, input_o,
                                     input_p), dim=1).float()
                    target = torch.cat((target_a, target_b, target_c, target_d, target_e, target_f, target_g, target_h,
                                     target_i, target_j, target_k, target_l, target_m, target_n, target_o,
                                     target_p), dim=1).float()



                    y, z, gate_reg, out_hidden, deltas = net.forward_n_step(obs_batch=inp, train_schedule=val_ss,
                                                                            predict_deltas=True,
                                                                            factor_delta=factor_output)

                    MSE = net.MSE(target, y).detach().item()

                    val_MSE_sum = val_MSE_sum + MSE

                    val_count = val_count + 1

                val_MSE_over_t[validations] = val_MSE_sum/val_count



                # Testing data:
                test_MSE_sum = 0.0
                test_gate_rate_sum = 0.0
                test_gate_dims_sum = 0.0
                test_gate_times_sum = 0.0
                test_count = 0
                for data_a, data_b, data_c, data_d, data_e, data_f, data_g, data_h, data_i, data_j, data_k, data_l,\
                    data_m, data_n, data_o, data_p in zip(*test_dataloaders):

                    input_a = data_a[0].permute(1, 0, 2)
                    target_a = data_a[1].permute(1, 0, 2)
                    input_b = data_b[0].permute(1, 0, 2)
                    target_b = data_b[1].permute(1, 0, 2)
                    input_c = data_c[0].permute(1, 0, 2)
                    target_c = data_c[1].permute(1, 0, 2)
                    input_d = data_d[0].permute(1, 0, 2)
                    target_d = data_d[1].permute(1, 0, 2)
                    input_e = data_e[0].permute(1, 0, 2)
                    target_e = data_e[1].permute(1, 0, 2)
                    input_f = data_f[0].permute(1, 0, 2)
                    target_f = data_f[1].permute(1, 0, 2)

                    input_g = data_g[0].permute(1, 0, 2)
                    target_g = data_g[1].permute(1, 0, 2)
                    input_h = data_h[0].permute(1, 0, 2)
                    target_h = data_h[1].permute(1, 0, 2)
                    input_i = data_i[0].permute(1, 0, 2)
                    target_i = data_i[1].permute(1, 0, 2)
                    input_j = data_j[0].permute(1, 0, 2)
                    target_j = data_j[1].permute(1, 0, 2)
                    input_k = data_k[0].permute(1, 0, 2)
                    target_k = data_k[1].permute(1, 0, 2)

                    input_l = data_l[0].permute(1, 0, 2)
                    target_l = data_l[1].permute(1, 0, 2)
                    input_m = data_m[0].permute(1, 0, 2)
                    target_m = data_m[1].permute(1, 0, 2)
                    input_n = data_n[0].permute(1, 0, 2)
                    target_n = data_n[1].permute(1, 0, 2)
                    input_o = data_o[0].permute(1, 0, 2)
                    target_o = data_o[1].permute(1, 0, 2)

                    input_p = data_p[0].permute(1, 0, 2)
                    target_p = data_p[1].permute(1, 0, 2)

                    inp = torch.cat((input_a, input_b, input_c, input_d, input_e, input_f, input_g, input_h, input_i,
                                     input_j, input_k, input_l, input_m, input_n, input_o,
                                     input_p), dim=1).float()
                    target = torch.cat((target_a, target_b, target_c, target_d, target_e, target_f, target_g, target_h,
                                     target_i, target_j, target_k, target_l, target_m, target_n, target_o,
                                     target_p), dim=1).float()


                    optimizer.zero_grad()
                    y, z, gate_reg, out_hidden, deltas = net.forward_n_step(obs_batch=inp, train_schedule=val_ss,
                                                                            predict_deltas=True,
                                                                            factor_delta=factor_output)

                    MSE = net.MSE(target, y).detach().item()

                    test_MSE_sum = test_MSE_sum + MSE

                    if network_name == 'GateL0RD':
                        # For GateL0RD we analyze the gating
                        gate_activity = gate_reg[0, :, :, :]
                        test_gate_rate_sum += calc_gate_rate(gate_activity)
                        test_gate_dims_sum += calc_gate_dims_used(gate_activity)
                        test_gate_times_sum += calc_gate_open_times(gate_activity)

                    test_count = test_count + 1
                test_MSE_over_t[validations] = test_MSE_sum / test_count
                test_gate_rate[validations] = test_gate_rate_sum/test_count
                test_gate_dims[validations] = test_gate_dims_sum/test_count
                test_gate_times[validations] = test_gate_times_sum/test_count

            validations += 1

            # Save the model after each validation
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

            # Training:
            for data_a, data_b, data_c, data_d, data_e, data_f, data_g, data_h, data_i, data_j, data_k, data_l, data_m, data_n, data_o, \
                data_p in zip(*train_dataloaders):


                input_a = data_a[0].permute(1, 0, 2)
                target_a = data_a[1].permute(1, 0, 2)
                input_b = data_b[0].permute(1, 0, 2)
                target_b = data_b[1].permute(1, 0, 2)
                input_c = data_c[0].permute(1, 0, 2)
                target_c = data_c[1].permute(1, 0, 2)
                input_d = data_d[0].permute(1, 0, 2)
                target_d = data_d[1].permute(1, 0, 2)
                input_e = data_e[0].permute(1, 0, 2)
                target_e = data_e[1].permute(1, 0, 2)
                input_f = data_f[0].permute(1, 0, 2)
                target_f = data_f[1].permute(1, 0, 2)

                input_g = data_g[0].permute(1, 0, 2)
                target_g = data_g[1].permute(1, 0, 2)
                input_h = data_h[0].permute(1, 0, 2)
                target_h = data_h[1].permute(1, 0, 2)
                input_i = data_i[0].permute(1, 0, 2)
                target_i = data_i[1].permute(1, 0, 2)
                input_j = data_j[0].permute(1, 0, 2)
                target_j = data_j[1].permute(1, 0, 2)
                input_k = data_k[0].permute(1, 0, 2)
                target_k = data_k[1].permute(1, 0, 2)

                input_l = data_l[0].permute(1, 0, 2)
                target_l = data_l[1].permute(1, 0, 2)
                input_m = data_m[0].permute(1, 0, 2)
                target_m = data_m[1].permute(1, 0, 2)
                input_n = data_n[0].permute(1, 0, 2)
                target_n = data_n[1].permute(1, 0, 2)
                input_o = data_o[0].permute(1, 0, 2)
                target_o = data_o[1].permute(1, 0, 2)

                input_p = data_p[0].permute(1, 0, 2)
                target_p = data_p[1].permute(1, 0, 2)

                inp = torch.cat((input_a, input_b, input_c, input_d, input_e, input_f, input_g, input_h, input_i,
                                 input_j, input_k, input_l, input_m, input_n, input_o,
                                 input_p), dim=1).float()
                target = torch.cat((target_a, target_b, target_c, target_d, target_e, target_f, target_g, target_h,
                                 target_i, target_j, target_k, target_l, target_m, target_n, target_o,
                                 target_p), dim=1).float()

                # Scheduled sampling
                ss = np.ones((seq_len, batch_size))
                if not teacherforcing:
                    if ss_slope == 0:
                        ss_epsilon = -1
                    else:
                        ss_epsilon = exponential_ss_prob(epoch=epoch, slope=ss_slope, min_value=0.0)
                    ss[start_steps:seq_len, :] = ss_epsilon

                optimizer.zero_grad()

                y, z, gate_reg, out_hidden, deltas = net.forward_n_step(obs_batch=inp, train_schedule=ss,
                                                                        predict_deltas=True,
                                                                        factor_delta=factor_output)

                loss = net.loss(y, target, gate_reg)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_value)
                optimizer.step()


    val_MSE_np_file = os.path.join(target_dir, "val_MSE_np")
    np.save(val_MSE_np_file, val_MSE_over_t)

    test_MSE_np_file = os.path.join(target_dir, "test_MSE_np")
    np.save(test_MSE_np_file, test_MSE_over_t)

    # Save gate analysis
    test_gate_rate_np_file = os.path.join(target_dir, "test_gate_rate_np.npy")
    np.save(test_gate_rate_np_file, test_gate_rate)
    test_gate_dims_np_file = os.path.join(target_dir, "test_gate_dims_np.npy")
    np.save(test_gate_dims_np_file, test_gate_dims)
    test_gate_times_np_file = os.path.join(target_dir, "test_gate_times_np.npy")
    np.save(test_gate_times_np_file, test_gate_times)

    dir_name_checkpoint = os.path.join(target_dir, "final_checkpoint")
    # Save a checkpoint
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validations': validations
    }, dir_name_checkpoint)

