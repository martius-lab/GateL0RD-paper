import torch
import torch.nn as nn
import numpy as np
import models.rnn_helpers as rnn_helpers
from models.gatel0rd_cell import GateL0RDCell

class GateL0RDModel(nn.Module):

    def __init__(self, input_dim, output_dim, latent_dim, feature_dim, reg_lambda, num_layers_internal, f_pre_layers,
                 f_post_layers, f_init_layers, f_init_inputs, num_layers=1, output_gate=True, stochastic_gates=True,
                 gate_noise_level=0.1, concat_gates=True, gate_type='ReTanh', l_loss=0):
        
        super(GateL0RDModel, self).__init__()


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers


        # f_pre network:
        self.fc_pre = rnn_helpers.create_f_pre(f_pre_layers=f_pre_layers, input_dim=input_dim, feature_dim=feature_dim)


        # f_init initializes the latent state (warm up)
        self.use_warm_up = f_init_layers > 0 and f_init_inputs > 0
        self.warm_up_inputs = f_init_inputs
        self.warm_up_network_list = nn.ModuleList([])
        if self.use_warm_up:
            for _ in range(self.num_layers):
                self.warm_up_network_list.append(rnn_helpers.create_f_init(f_init_layers=f_init_layers,
                                                                           f_init_inputs=f_init_inputs,
                                                                           input_dim=input_dim, feature_dim=feature_dim,
                                                                           latent_dim=latent_dim))


        # The RNN core: (stacked) GateL0RD cells:
        self.cells = nn.ModuleList([])
        for _ in range(num_layers):
            self.cells.append(GateL0RDCell(input_dim=feature_dim, output_dim=feature_dim, latent_dim=latent_dim,
                                           num_layers_internal=num_layers_internal, stochastic_gates=stochastic_gates,
                                           gate_noise_level=gate_noise_level, concat_gates=concat_gates,
                                           gate_type=gate_type, output_gate=output_gate))

        # f_post network
        self.fc_out = rnn_helpers.create_f_post(f_post_layers=f_post_layers, feature_dim=feature_dim, output_dim=output_dim)

        # L_0-Gates:
        self.reg_lambda = reg_lambda
        assert 0 <= l_loss <= 2 and isinstance(l_loss, int), "Only L_{0,1,2} norm implemented, got L_" + str(l_loss) + " norm."
        assert l_loss == 0 or gate_type == 'Sigmoid', 'L_' + str(l_loss) + ' norm needs sigmoidal gates.'
        self.L_loss = l_loss

        self.MSE = nn.MSELoss()


    def forward_n_step(self, obs_batch, train_schedule, predict_deltas, factor_delta=-1, action_batch=None):
        """
        Forward prediction in autoregressive mode
        :param obs_batch: batch of observations (shape: sequence length  X batch size X observation dim)
        :param train_schedule: scheduled sampling, determines probability for using real input (np array, shape: sequence length  X batch size)
        :param predict_deltas: states whether network predicts x_t+1 or \Delta x_t+1
        :param factor_delta: when predicting deltas one can specifiy a constant k for which the network predicts k * \Delta x_t+1
        :param action_batch: batch of actions for POMDPs (shape: sequence length  X batch size X action dim)
        :return: - batch of network outputs(shape: sequence length  X batch size X output dim)
                 - latent states (shape: layer number x sequence length  X batch size X latent dim)
                 - regularized gate activations \Theta (shape: layer number x sequence length  X batch size X latent dim)
                 - layer outputs (shape: layer number x sequence length  X batch size X feature dim)
                 - network outputs as deltas (shape: sequence length  X batch size X output dim)
        """

        seq_len, batch_size, obs_dim = obs_batch.size()

        # Sample based on scheduled sampling probabilities
        sampling_rand = np.random.rand(seq_len, batch_size)
        sampling_schedule_np = np.clip(np.ceil(train_schedule - sampling_rand), 0, 1)
        sampling_schedule = torch.unsqueeze(torch.from_numpy(sampling_schedule_np), dim=2).expand(seq_len, batch_size, obs_dim).float()


        # If actions are specified we are in planning mode
        forward_for_planning = action_batch is not None


        # Keep track of all latent states (hs), layer outputs (ys) regularized gate activations
        # (gs_reg, i.e. Theta(s) in paper), network outputs (outs or delta_outs):
        last_out = None
        hs = []
        ys = []
        gs_reg = []
        outs = []
        delta_outs = []


        for t in range(seq_len):

            hs_at_t = []
            ys_at_t = []
            gs_reg_at_t = []

            # Determine network based on scheduled sampling
            if last_out is None:
                x_in_t = obs_batch[t, :, :]
            else:
                x_in_t = sampling_schedule[t, :, :] *  obs_batch[t, :, :] + (1 - sampling_schedule[t, :, :]) * last_out


            # For planning the actions need to be concatenated
            if forward_for_planning:
                x_in_t_full = torch.cat((action_batch[t, :, :], x_in_t), 1)
            else:
                x_in_t_full = x_in_t

            # Preprocessing of input
            x_lt = self.fc_pre(x_in_t_full)

            # Propagate the input through all layers
            for layer in range(self.num_layers):

                # Get the last latent state
                if t > 0:
                    # There exists a previous latent state
                    last_h_l = hs[t-1][layer, :, :]
                else:
                    # Initialize the latent state, either through warm up or with zeroes
                    if self.use_warm_up:
                        last_h_l = self.__warm_up(obs_batch=obs_batch, layer=layer, action_batch=action_batch)
                    else:
                        last_h_l = torch.zeros((batch_size, self.latent_dim))

                y_lt, h_lt, g_lt_reg = self.__forward_per_layer(x_lt=x_lt, h_ltminus1=last_h_l, l=layer)

                # Input to the next layer is the output of the current layer
                x_lt = y_lt

                # Store all latent states and gate activations
                hs_at_t.append(h_lt)
                ys_at_t.append(y_lt)
                gs_reg_at_t.append(g_lt_reg)


            # Stack the latent states and gate activations of all layers
            hs.append(torch.stack(hs_at_t))
            ys.append(torch.stack(ys_at_t))
            gs_reg.append(torch.stack(gs_reg_at_t))

            # Use the read-out layer to compute the outputs
            out_pre = self.fc_out(x_lt)

            # Save the network outputs (and deltas)
            delta_outs.append(out_pre)
            last_out = out_pre
            if predict_deltas:
                last_out = x_in_t + factor_delta * out_pre
            outs.append(last_out)

        # Stack the latent states, gate activations, etc... over all time steps
        # We move the layer-dimension to the front, since its 1 in most cases
        hs_final = torch.stack(hs).permute([1, 0, 2, 3])
        ys_final = torch.stack(ys).permute([1, 0, 2, 3])
        gs_reg_final = torch.stack(gs_reg).permute([1, 0, 2, 3])

        return torch.stack(outs), hs_final, gs_reg_final, ys_final, torch.stack(delta_outs)


    def __warm_up(self, obs_batch, layer, action_batch= None):
        """
        Implements f_init call
        :param obs_batch: batch of observations (shape: sequence length  X batch size X observation dim)
        :param layer: index l of layer when using a stacked GateL0RD
        :param action_batch: batch of actions for POMDPs (shape: sequence length  X batch size X action dim)
        :return: h_0 for layer l
        """

        seq_len, _, _ = obs_batch.size()
        forward_for_planning = action_batch is not None

        # Concatenate the first inputs used to warm up the latent state
        if forward_for_planning:
            warm_up_input = torch.cat((action_batch[0, :, :], obs_batch[0, :, :]), 1)
        else:
            warm_up_input = obs_batch[0, :, :]

        for t_warm_up in range(self.warm_up_inputs - 1):
            next_warm_up_input = obs_batch[t_warm_up + 1, :, :]
            if forward_for_planning:
                warm_up_input_t_plus1 = torch.cat((action_batch[t_warm_up + 1, :, :], next_warm_up_input), 1)
            else:
                warm_up_input_t_plus1 = next_warm_up_input
            warm_up_input = torch.cat((warm_up_input, warm_up_input_t_plus1), 1)

        return self.warm_up_network_list[layer](warm_up_input)



    def forward_one_step(self, obs_t, predict_deltas, factor_delta, action_t=None, h_tminus1=None):
        """
        Forward prediction in for one step
        :param obs_t: batch of observations (shape: batch size X observation dim)
        :param predict_deltas: states whether network predicts x_t+1 or \Delta x_t+1
        :param factor_delta: when predicting deltas one can specifiy a constant k for which the network predicts k * \Delta x_t+1
        :param action_t: batch of actions for POMDPs (shape: batch size X action dim)
        :param h_tminus1: last latent state (shape: layer number X batch size X action dim)
        :return: - batch of network outputs(shape: batch size X output dim)
                 - latent states (shape: layer number  X batch size X latent dim)
                 - binary gate activations \Theta (shape: layer number  X batch size X latent dim)
                 - layer outputs (shape: layer number X batch size X feature dim)
                 - network outputs as deltas (shape: batch size X output dim)
        """

        if action_t is not None:
            input_t = torch.cat((action_t, obs_t), dim=1)
        else:
            input_t = obs_t

        B, _ = obs_t.shape

        list_of_hs = []
        list_of_gs_reg = []
        list_of_ys = []


        x = self.fc_pre(input_t)

        for l in range(self.num_layers):
            if h_tminus1 is None:
                if self.use_warm_up:
                    warm_up_actions = action_t
                    if action_t is not None:
                        warm_up_actions = torch.unsqueeze(action_t, 0)
                    last_h_l = self.__warm_up(obs_batch=torch.unsqueeze(obs_t, 0), layer=l, action_batch=warm_up_actions)
                else:
                    last_h_l = torch.zeros((B, self.latent_dim))

            else:
                last_h_l = h_tminus1[l, :, :]

            y_lt, h_lt, g_lt_reg, g_lt = self.__forward_per_layer(x_lt=x, h_ltminus1=last_h_l, l=l)

            list_of_hs.append(h_lt)
            list_of_gs_reg.append(g_lt_reg)
            list_of_ys.append(y_lt)
            x = y_lt

        hs = torch.stack(list_of_hs)
        g_regs = torch.stack(list_of_gs_reg)
        ys = torch.stack(list_of_ys)

        out_pre = self.fc_out(x)
        delta_out = out_pre
        net_out = out_pre
        if predict_deltas:
            net_out = obs_t + factor_delta * out_pre

        return net_out, hs, g_regs, ys, delta_out



    def __forward_per_layer(self, x_lt, h_ltminus1, l):
        """
        Forward pass one step through one layer of GateL0RD, i.e. pass through g-, r-, p- and o-network of one cell
        """
        return self.cells[l](x_lt, h_ltminus1)


    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-04)

    def loss(self, out, tar, g_regs):
        gate_loss = torch.mean(g_regs)
        if self.L_loss == 2:
            gate_loss = gate_loss**2
        return self.MSE(out, tar) + self.reg_lambda * gate_loss
