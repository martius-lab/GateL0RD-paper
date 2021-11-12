import argparse
import sys
import smart_settings
import os
import numpy as np
import torch
import datasets.shepherd_dataloader as shepherd_dataloader
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

    # General dimensionality of the scenario
    input_dim = 10
    output_dim = 7
    action_dim = 3
    factor_output = 0.1

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
    dataset_split_rs = params.dataset_split_rs
    num_data_train = 3200
    num_data_val = 1600
    num_data_test = 3200

    seq_len = 100
    subdataset_batch_size = 32
    num_subdatasets = 4
    batch_size = subdataset_batch_size * num_subdatasets
    # For computing the sheep appearance error we only take 3 datsets because in one the sheep remains hidden
    sheep_error_batch_size = subdataset_batch_size * 3

    dataloaders_train, dataloaders_val, dataloaders_test = shepherd_dataloader.get_shepherd_dataloaders(dataset_split_rs=dataset_split_rs,
                                                                                                    subdataset_batch_size=subdataset_batch_size,
                                                                                                    factor_output=factor_output,
                                                                                                    num_data_train=num_data_train,
                                                                                                    num_data_val=num_data_val,
                                                                                                    num_data_test=num_data_test)
    # Scheduled Sampling
    ss_slope = params.ss_slope
    start_steps = params.start_steps

    # Gradient norm clipping
    grad_clip = False
    grad_clip_value = 1000.0
    if params.grad_clip_value > 0:
        grad_clip_value = params.grad_clip_value
        grad_clip = True

    t_validation = 100
    num_validations = 10 * params.train_len
    num_epochs = t_validation * num_validations

    val_MSE_over_t = np.zeros(num_validations, dtype='float64')
    val_sheep_MSE_over_t = np.zeros(num_validations, dtype='float64')
    test_MSE_over_t = np.zeros(num_validations, dtype='float64')
    test_sheep_MSE_over_t = np.zeros(num_validations, dtype='float64')
    loss_over_t = np.zeros(num_epochs, dtype='float64')

    epoch_start = 0
    validations = 0

    for epoch in range(epoch_start, num_epochs):
        # Log one plot for every 1000th epoch
        if epoch % t_validation == 0:
            net = net.eval()

            #No scheduled sampling during validation
            val_ss = np.ones((seq_len, batch_size))
            val_ss[start_steps:seq_len, :] = -1

            with torch.no_grad():

                # Validation
                val_MSE_sum = 0.0
                val_count = 0
                for data_a, data_b, data_c, data_d in zip(*dataloaders_val):

                    # Move sequence length to the front
                    input_a = data_a[0].permute(1, 0, 2)
                    input_b = data_b[0].permute(1, 0, 2)
                    input_c = data_c[0].permute(1, 0, 2)
                    input_d = data_d[0].permute(1, 0, 2)

                    target_a = data_a[1].permute(1, 0, 2)
                    target_b = data_b[1].permute(1, 0, 2)
                    target_c = data_c[1].permute(1, 0, 2)
                    target_d = data_d[1].permute(1, 0, 2)

                    inp = torch.cat((input_a, input_b, input_c, input_d), dim=1).float()
                    target = torch.cat((target_a, target_b, target_c, target_d), dim=1).float()

                    actions = inp[:, :, 0:action_dim]
                    states = inp[:, :, action_dim:input_dim]


                    y, z, gate_reg, out_hidden, _ = net.forward_n_step(obs_batch=states, train_schedule=val_ss,
                                                                              predict_deltas=True,
                                                                              factor_delta=factor_output,
                                                                              action_batch=actions)

                    real_next_states = states + (factor_output * target)
                    pred_next_states = y

                    MSE = net.MSE(real_next_states, pred_next_states).detach().item()
                    val_MSE_sum = val_MSE_sum + MSE
                    val_count = val_count + 1
                val_MSE_over_t[validations] = val_MSE_sum / val_count


                val_MSE_x_appear_sum = 0.0
                val_count = 0
                for _, data_a, data_b, data_c in zip(*dataloaders_val):
                    # Only 3 datasets considered because in one the gate is not opened

                    input_a = data_a[0].permute(1, 0, 2)
                    input_b = data_b[0].permute(1, 0, 2)
                    input_c = data_c[0].permute(1, 0, 2)

                    target_a = data_a[1].permute(1, 0, 2)
                    target_b = data_b[1].permute(1, 0, 2)
                    target_c = data_c[1].permute(1, 0, 2)

                    info_a = data_a[2].permute(1, 0, 2)
                    info_b = data_b[2].permute(1, 0, 2)
                    info_c = data_c[2].permute(1, 0, 2)

                    inf = torch.cat((info_a, info_b, info_c), dim=1).float()
                    inp = torch.cat((input_a, input_b, input_c), dim=1).float()
                    target = torch.cat((target_a, target_b, target_c), dim=1).float()

                    inf_np = inf.detach().numpy()

                    actions = inp[:, :, 0:action_dim]
                    states = inp[:, :, action_dim:input_dim]

                    # Teacher forcing up to point of gate opening

                    new_ss = np.ones((seq_len, sheep_error_batch_size))
                    new_ss[start_steps:seq_len, :] = 1 - inf_np[start_steps:, :, 0]

                    overall_outputs = seq_len * sheep_error_batch_size
                    non_masked_outputs = torch.sum(inf[:, :, 0]).detach().item()

                    y, z, gate_reg, out_hidden, _ = net.forward_n_step(obs_batch=states, train_schedule=new_ss,
                                                                              predict_deltas=True,
                                                                              factor_delta=factor_output,
                                                                              action_batch=actions)


                    real_next_states = states + (factor_output * target)
                    pred_next_states = y

                    # Mask out errors except for time step of gate opening
                    error_mask_appearance = torch.zeros((seq_len, sheep_error_batch_size, 2))
                    delta_inf = (inf[1:seq_len, :, 0:1] - inf[0:(seq_len - 1), :, 0:1]).expand(seq_len - 1, sheep_error_batch_size, 2)
                    error_mask_appearance[1:, :, :] = delta_inf[:, :, :]
                    MSE_x_appear = net.MSE(real_next_states[:, :, 4:5] * error_mask_appearance, pred_next_states[:, :,4:5] * error_mask_appearance).detach().item() * overall_outputs / sheep_error_batch_size
                    val_MSE_x_appear_sum += MSE_x_appear
                    val_count = val_count + 1
                val_sheep_MSE_over_t[validations] = val_MSE_x_appear_sum / val_count

                # Testing
                test_MSE_sum = 0.0
                test_count = 0
                for data_a, data_b, data_c, data_d in zip(*dataloaders_test):

                    input_a = data_a[0].permute(1, 0, 2)
                    input_b = data_b[0].permute(1, 0, 2)
                    input_c = data_c[0].permute(1, 0, 2)
                    input_d = data_d[0].permute(1, 0, 2)

                    target_a = data_a[1].permute(1, 0, 2)
                    target_b = data_b[1].permute(1, 0, 2)
                    target_c = data_c[1].permute(1, 0, 2)
                    target_d = data_d[1].permute(1, 0, 2)


                    inp = torch.cat((input_a, input_b, input_c, input_d), dim=1).float()
                    target = torch.cat((target_a, target_b, target_c, target_d), dim=1).float()

                    actions = inp[:, :, 0:action_dim]
                    states = inp[:, :, action_dim:input_dim]

                    y, z, gate_reg, out_hidden, _ = net.forward_n_step(obs_batch=states, train_schedule=val_ss,
                                                                              predict_deltas=True,
                                                                              factor_delta=factor_output,
                                                                              action_batch=actions)

                    real_next_states = states + (factor_output * target)
                    pred_next_states = y

                    MSE = net.MSE(real_next_states, pred_next_states).detach().item()
                    test_MSE_sum = test_MSE_sum + MSE
                    test_count = test_count + 1

                test_MSE_over_t[validations] = test_MSE_sum / test_count

                test_MSE_x_appear_sum = 0.0
                test_count = 0
                for _, data_a, data_b, data_c in zip(*dataloaders_test):
                    input_a = data_a[0].permute(1, 0, 2)
                    input_b = data_b[0].permute(1, 0, 2)
                    input_c = data_c[0].permute(1, 0, 2)

                    target_a = data_a[1].permute(1, 0, 2)
                    target_b = data_b[1].permute(1, 0, 2)
                    target_c = data_c[1].permute(1, 0, 2)

                    info_a = data_a[2].permute(1, 0, 2)
                    info_b = data_b[2].permute(1, 0, 2)
                    info_c = data_c[2].permute(1, 0, 2)

                    inf = torch.cat((info_a, info_b, info_c), dim=1).float()
                    inp = torch.cat((input_a, input_b, input_c), dim=1).float()
                    target = torch.cat((target_a, target_b, target_c), dim=1).float()

                    inf_np = inf.detach().numpy()

                    actions = inp[:, :, 0:action_dim]
                    states = inp[:, :, action_dim:input_dim]

                    # Teacher forcing up to point of gate opening
                    new_ss = np.ones((seq_len, inp.shape[1]))
                    new_ss[start_steps:seq_len, :] = 1 - inf_np[start_steps:, :, 0]

                    overall_outputs = seq_len * sheep_error_batch_size
                    non_masked_outputs = torch.sum(inf[:, :, 0]).detach().item()

                    y, z, gate_reg, out_hidden, _ = net.forward_n_step(obs_batch=states, train_schedule=new_ss,
                                                                              predict_deltas=True,
                                                                              factor_delta=factor_output,
                                                                              action_batch=actions)

                    real_next_states = states + (factor_output * target)
                    pred_next_states = y

                    # Mask out errors except for time step of gate opening
                    error_mask_appearance = torch.zeros((seq_len, sheep_error_batch_size, 2))
                    delta_inf = (inf[1:seq_len, :, 0:1] - inf[0:(seq_len - 1), :, 0:1]).expand(seq_len - 1, sheep_error_batch_size, 2)
                    error_mask_appearance[1:, :, :] = delta_inf[:, :, :]
                    MSE_x_appear = net.MSE(real_next_states[:, :, 4:5] * error_mask_appearance, pred_next_states[:, :,4:5] * error_mask_appearance).detach().item() * overall_outputs/sheep_error_batch_size
                    test_MSE_x_appear_sum += MSE_x_appear
                    test_count = test_count + 1
                test_sheep_MSE_over_t[validations] = test_MSE_x_appear_sum / test_count

            validations += 1

            dir_name_checkpoint = os.path.join(target_dir, "checkpoint_v" + str(validations))
            # Save a checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validations': validations
            }, dir_name_checkpoint)



        # training:
        net = net.train()
        train_count = 0
        mean_loss = 0.0
        for data_a, data_b, data_c, data_d in zip(*dataloaders_train):

            input_a = data_a[0].permute(1, 0, 2)
            input_b = data_b[0].permute(1, 0, 2)
            input_c = data_c[0].permute(1, 0, 2)
            input_d = data_d[0].permute(1, 0, 2)

            target_a = data_a[1].permute(1, 0, 2)
            target_b = data_b[1].permute(1, 0, 2)
            target_c = data_c[1].permute(1, 0, 2)
            target_d = data_d[1].permute(1, 0, 2)

            inp = torch.cat((input_a, input_b, input_c, input_d), dim=1).float()
            target = torch.cat((target_a, target_b, target_c, target_d), dim=1).float()

            actions = inp[:, :, 0:action_dim]
            states = inp[:, :, action_dim:input_dim]

            # Scheduled sampling
            ss = np.ones((seq_len, batch_size))
            if ss_slope == 0:
                ss_epsilon = 0.0
            else:
                ss_epsilon = exponential_ss_prob(epoch=epoch, slope=ss_slope, min_value=0.05)
            ss[start_steps:seq_len, :] = ss_epsilon

            optimizer.zero_grad()
            y, z, gate_reg, out_hidden, _ = net.forward_n_step(obs_batch=states, train_schedule=ss,
                                                                      predict_deltas=True, factor_delta=factor_output,
                                                                      action_batch=actions)

            real_next_states = states + (factor_output * target)
            pred_next_states = y

            loss = net.loss(pred_next_states, real_next_states, gate_reg)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_value)
            optimizer.step()

            train_count += 1
            mean_loss += loss.detach().item()
        loss_over_t[epoch] = mean_loss / train_count


    # Save MSE arrays
    val_MSE_np_file = os.path.join(target_dir, "val_MSE_np")
    np.save(val_MSE_np_file, val_MSE_over_t)

    test_MSE_np_file = os.path.join(target_dir, "test_MSE_np")
    np.save(test_MSE_np_file, test_MSE_over_t)

    val_sheep_MSE_np_file = os.path.join(target_dir, "val_sheep_MSE_np")
    np.save(val_sheep_MSE_np_file, val_sheep_MSE_over_t)

    test_sheep_MSE_np_file = os.path.join(target_dir, "test_sheep_MSE_np")
    np.save(test_sheep_MSE_np_file, test_sheep_MSE_over_t)

    loss_np_file = os.path.join(target_dir, "loss_np")
    np.save(loss_np_file, loss_over_t)

    dir_name_checkpoint = os.path.join(target_dir, "final_checkpoint")
    # Save a checkpoint
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validations': validations
    }, dir_name_checkpoint)

