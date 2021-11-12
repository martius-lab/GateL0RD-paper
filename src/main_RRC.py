import argparse
import sys
import smart_settings
import os
import numpy as np
import torch
from datasets.rrc_dataset import RRCDataset
import models.gatel0rd_model as GateL0RD
import models.rnn_model as RNNs
from utils.scheduled_sampling_helper import exponential_ss_prob
from utils.set_rs import set_rs


parser = argparse.ArgumentParser()
parser.add_argument("-config", help="Path to configuration file")
parser.add_argument("-seed", type=int, help="Seed for RNG")



def RRC_validation(control_dataloader, nocontrol_dataloader, network,
                   seq_len, batch_size, start_steps, action_dim,
                   input_dim, factor_output):

    network = network.eval()

    # Scheduled sampling not used during validation
    val_ss = np.ones((seq_len, batch_size))
    val_ss[start_steps:seq_len, :] = -1

    val_MSE_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for data_a, data_b in zip(control_dataloader, nocontrol_dataloader):

            # Sequence_length first
            input_a = data_a[0].permute(1, 0, 2)
            target_a = data_a[1].permute(1, 0, 2)
            input_b = data_b[0].permute(1, 0, 2)
            target_b = data_b[1].permute(1, 0, 2)

            inp = torch.cat((input_a, input_b), dim=1).float()
            target = torch.cat((target_a, target_b), dim=1).float()

            actions = inp[:, :, 0:action_dim]
            states = inp[:, :, action_dim:input_dim]

            y, z, gate_reg, out_hidden, _ = network.forward_n_step(obs_batch=states, train_schedule=val_ss,
                                                                   predict_deltas=True, factor_delta=factor_output,
                                                                   action_batch=actions)

        real_next_states = states + (factor_output * target)
        pred_next_states = y

        MSE = network.MSE(real_next_states, pred_next_states).detach().item()

        val_MSE_sum = val_MSE_sum + MSE
        val_count = val_count + 1
    return val_MSE_sum/val_count




if __name__ == '__main__':

    args = parser.parse_args(sys.argv[1:])
    params = smart_settings.load(args.config)

    # Random seed
    if args.seed is not None:
        rand_seed = args.seed
    else:
        rand_seed = params.rs
    set_rs(rand_seed)

    # Target directory:
    target_dir = params.model_dir + str(rand_seed) + '/'
    os.makedirs(target_dir, exist_ok=True)

    # RNN type
    network_name = params.network_name

    num_data_train = 3200
    num_data_val = 3200
    input_dim = 6
    output_dim = 4
    info_dim = 9
    action_dim = 2
    factor_output = 0.1
    seq_len = 49

    if network_name == 'GateL0RD':
        net = GateL0RD.GateL0RDModel(input_dim=input_dim, output_dim=output_dim, latent_dim=params.latent_dim,
                                     feature_dim=params.feature_dim, num_layers=params.layer_num,
                                     num_layers_internal=params.num_layers_internal, reg_lambda=params.reg_lambda,
                                     f_pre_layers=params.preprocessing_layers,
                                     f_post_layers=params.postprocessing_layers,
                                     f_init_layers=params.warm_up_layers, f_init_inputs=params.warm_up_inputs,
                                     stochastic_gates=params.stochastic_gates,
                                     gate_noise_level=params.gate_noise_level, gate_type=params.gate_type,
                                     output_gate=params.output_gate, l_loss=params.l_loss)
    else:
        net = RNNs.RNNModel(input_dim=input_dim, output_dim=output_dim, latent_dim=params.latent_dim,
                               feature_dim=params.feature_dim, num_layers=params.layer_num,
                               rnn_type=network_name, f_pre_layers=params.preprocessing_layers,
                               f_post_layers=params.postprocessing_layers, f_init_layers=params.warm_up_layers,
                               f_init_inputs=params.warm_up_inputs)
    optimizer = net.get_optimizer(params.lr)


    data_path = 'data/RRC/' + params.train_data_type
    gen_data_path = 'data/RRC/Generalization'

    # We create two datasets for every setting (training, validation, testing, generalization) containing an equal number
    # of samples in which the robot was controlled and not

    # Training:
    dataset_train1 = RRCDataset(path=data_path, num_data=num_data_train, control=True, train=True,
                                output_factor=factor_output, index_offset=0)
    dataset_train2 = RRCDataset(path=data_path, num_data=num_data_train, control=False, train=True,
                                output_factor=factor_output, index_offset=0)

    # Validation
    dataset_val1 = RRCDataset(path=data_path, num_data=num_data_val, control=True, train=False,
                              output_factor=factor_output, index_offset=0)
    dataset_val2 = RRCDataset(path=data_path, num_data=num_data_val, control=False, train=False,
                              output_factor=factor_output, index_offset=0)

    # Validation generalization
    dataset_val1_gen = RRCDataset(path=gen_data_path, num_data=num_data_val, control=True, train=False,
                                  output_factor=factor_output, index_offset=0)
    dataset_val2_gen = RRCDataset(path=gen_data_path, num_data=num_data_val, control=False, train=False,
                                  output_factor=factor_output, index_offset=0)
    # Testing
    dataset_test1 = RRCDataset(path=data_path, num_data=num_data_val, control=True, train=False,
                               output_factor=factor_output, index_offset=num_data_val)
    dataset_test2 = RRCDataset(path=data_path, num_data=num_data_val, control=False, train=False,
                               output_factor=factor_output, index_offset=num_data_val)

    # Testing generalization
    dataset_test1_gen = RRCDataset(path=gen_data_path, num_data=num_data_val, control=True, train=False,
                                   output_factor=factor_output, index_offset=num_data_val)
    dataset_test2_gen = RRCDataset(path=gen_data_path, num_data=num_data_val, control=False, train=False,
                                   output_factor=factor_output, index_offset=num_data_val)


    subdataset_batch_size = 64
    num_subdatasets = 2
    batch_size = subdataset_batch_size * num_subdatasets

    # Training
    dataloader_control = torch.utils.data.DataLoader(dataset_train1, batch_size=subdataset_batch_size, shuffle=True)
    dataloader_no_control = torch.utils.data.DataLoader(dataset_train2, batch_size=subdataset_batch_size, shuffle=True)

    # Validation:
    val_dataloader_control = torch.utils.data.DataLoader(dataset_val1, batch_size=subdataset_batch_size, shuffle=True)
    val_dataloader_no_control = torch.utils.data.DataLoader(dataset_val2, batch_size=subdataset_batch_size,
                                                            shuffle=True)

    # Generalization validation
    val_dataloader_control_gen = torch.utils.data.DataLoader(dataset_val1_gen,
                                                                  batch_size=subdataset_batch_size, shuffle=True)
    val_dataloader_no_control_gen = torch.utils.data.DataLoader(dataset_val2_gen,
                                                                     batch_size=subdataset_batch_size, shuffle=True)

    # Testing:
    test_dataloader_control = torch.utils.data.DataLoader(dataset_test1, batch_size=subdataset_batch_size, shuffle=True)
    test_dataloader_no_control = torch.utils.data.DataLoader(dataset_test2, batch_size=subdataset_batch_size,
                                                             shuffle=True)

    # Generalization testing
    test_dataloader_control_gen = torch.utils.data.DataLoader(dataset_test1_gen,
                                                                   batch_size=subdataset_batch_size,
                                                                   shuffle=True)
    test_dataloader_no_control_gen = torch.utils.data.DataLoader(dataset_test2_gen,
                                                                      batch_size=subdataset_batch_size,
                                                                      shuffle=True)

    # Scheduled Sampling
    ss_slope = params.ss_slope
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


    val_MSE_over_t = np.zeros(num_validations, dtype='float64')
    val_MSE_over_t_gen = np.zeros(num_validations, dtype='float64')

    test_MSE_over_t = np.zeros(num_validations, dtype='float64')
    test_MSE_over_t_gen = np.zeros(num_validations, dtype='float64')

    loss_over_t = np.zeros(num_epochs, dtype='float64')

    validations = 0
    epoch_start = 0

    for epoch in range(epoch_start, num_epochs):

        if epoch % t_validation == 0:

            # Normal validation
            val_MSE_over_t[validations] = RRC_validation(control_dataloader=val_dataloader_control, nocontrol_dataloader=val_dataloader_no_control,
                                                         network=net, seq_len=seq_len, batch_size=batch_size, start_steps=start_steps, action_dim=action_dim,
                                                         input_dim=input_dim, factor_output=factor_output)

            # Generalization validation
            val_MSE_over_t_gen[validations] = RRC_validation(control_dataloader=val_dataloader_control_gen,
                                                             nocontrol_dataloader=val_dataloader_no_control_gen,
                                                             network=net, seq_len=seq_len, batch_size=batch_size,
                                                             start_steps=start_steps, action_dim=action_dim,
                                                             input_dim=input_dim, factor_output=factor_output)

            # Normal testing
            test_MSE_over_t[validations] = RRC_validation(control_dataloader=test_dataloader_control,
                                                         nocontrol_dataloader=test_dataloader_no_control,
                                                         network=net, seq_len=seq_len, batch_size=batch_size,
                                                         start_steps=start_steps, action_dim=action_dim,
                                                         input_dim=input_dim, factor_output=factor_output)

            # Generalization testing
            test_MSE_over_t_gen[validations] = RRC_validation(control_dataloader=test_dataloader_control_gen,
                                                             nocontrol_dataloader=test_dataloader_no_control_gen,
                                                             network=net, seq_len=seq_len, batch_size=batch_size,
                                                             start_steps=start_steps, action_dim=action_dim,
                                                             input_dim=input_dim, factor_output=factor_output)

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

        net = net.train()


        train_count = 0
        mean_loss = 0.0
        # TRAINING:
        for data_a, data_b in zip(dataloader_control, dataloader_no_control):

            input_a = data_a[0].permute(1, 0, 2)
            target_a = data_a[1].permute(1, 0, 2)

            input_b = data_b[0].permute(1, 0, 2)
            target_b = data_b[1].permute(1, 0, 2)

            inp = torch.cat((input_a, input_b), dim=1)
            target = torch.cat((target_a, target_b), dim=1)


            actions = inp[:, :, 0:action_dim]
            states = inp[:, :, action_dim:input_dim]

            ss = np.ones((seq_len, batch_size))
            if not teacherforcing:
                if ss_slope == 0:
                    ss_epsilon = 0.0
                else:
                    ss_epsilon = exponential_ss_prob(epoch=epoch, slope=ss_slope, min_value=0.02)
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
        loss_over_t[epoch] = mean_loss/train_count


    val_MSE_np_file = os.path.join(target_dir, "val_MSE_np")
    np.save(val_MSE_np_file, val_MSE_over_t)

    test_MSE_np_file = os.path.join(target_dir, "test_MSE_np")
    np.save(test_MSE_np_file, test_MSE_over_t)

    val_MSE_gen_np_file = os.path.join(target_dir, "val_MSE_gen_np")
    np.save(val_MSE_gen_np_file, val_MSE_over_t_gen)

    test_MSE_gen_np_file = os.path.join(target_dir, "test_MSE_gen_np")
    np.save(test_MSE_gen_np_file, test_MSE_over_t_gen)

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

