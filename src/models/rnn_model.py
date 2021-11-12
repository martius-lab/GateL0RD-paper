import torch
import torch.nn as nn
import numpy as np
import models.rnn_helpers as rnn_helpers


class RNNModel(nn.Module):

    def __init__(self, input_dim, output_dim, latent_dim, feature_dim, num_layers, rnn_type,
                 f_pre_layers, f_post_layers,  f_init_layers, f_init_inputs):

        super(RNNModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        # preprocessing layers
        self.fc_in = rnn_helpers.create_f_pre(f_pre_layers=f_pre_layers, input_dim=input_dim, feature_dim=feature_dim)

        input_net = feature_dim

        # create the rnn core
        if rnn_type == 'ElmanRNN':
            self.rnn = nn.RNN(input_size=input_net, hidden_size=latent_dim, num_layers=num_layers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_net, hidden_size=latent_dim, num_layers=num_layers)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_net, hidden_size=latent_dim, num_layers=num_layers)
        self.rnn_type = rnn_type

        # f_init initializes the latent state (warm up)
        self.use_warm_up = f_init_layers > 0 and f_init_inputs > 0
        self.warm_up_inputs = f_init_inputs
        self.warm_up_network_list = nn.ModuleList([])
        if self.use_warm_up:
            for l in range(self.num_layers):
                latent_dim_factor = 1
                if rnn_type == 'LSTM':
                    # LSTM need cell state and hidden state to be 'warmed up'
                    latent_dim_factor = 2
                self.warm_up_network_list.append(rnn_helpers.create_f_init(f_init_layers=f_init_layers,
                                                                           f_init_inputs=f_init_inputs,
                                                                           input_dim=input_dim, feature_dim=feature_dim,
                                                                           latent_dim=int(latent_dim_factor * latent_dim)))
        # postprocessing layers
        self.fc_out = rnn_helpers.create_f_post(f_post_layers=f_post_layers, feature_dim=latent_dim, output_dim=output_dim)

        # loss
        self.MSE = nn.MSELoss()


    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-04)


    def loss(self, out, tar, gs):
        # gs are ignored, included to have same function call as GateL0RD
        return self.MSE(out, tar)


    def forward_one_step(self, obs_t, predict_deltas, factor_delta, action_t=None, h_tminus1=None, c_tminus1=None):

        if action_t is not None:
            input_t = torch.cat((action_t, obs_t), dim=1)
        else:
            input_t = obs_t

        input_batch = torch.unsqueeze(input_t, 0)

        if c_tminus1 is None and h_tminus1 is None:
            # Warm up:
            if self.use_warm_up:
                if self.rnn_type == 'LSTM':
                    last_hc = self.__warm_up_lstm(input_batch)
                else:
                    last_hc = self.__warm_up(input_batch)
            else:
                last_hc = None
        else:
            # One of the latent vectors is not None
            if self.rnn_type == 'LSTM':
                assert h_tminus1 is not None and c_tminus1 is not None
                last_hc = (h_tminus1, c_tminus1)
            else:
                last_hc = h_tminus1

        y, h, c = self.__forward(input_batch, last_hc)
        out = y[0, :, :]
        if predict_deltas:
            out = obs_t + factor_delta * y[0, :, :]
        return out, h, c, None, y[0,:, :] # 5 outputs to match GateL0RDModel


    def __forward(self, x, h=None):
        x_processed = self.fc_in(x)

        c = None
        if self.rnn_type == 'LSTM':
            y, (h, c) = self.rnn(x_processed, h)
        else:
            y, h = self.rnn(x_processed, h)
        y2 = self.fc_out(y)
        return y2, h, c

    def __warm_up(self, input_batch):
        warm_up_list = []
        for l in range(self.num_layers):
            warm_up_input = input_batch[0, :, :]
            for t_hat in range(self.warm_up_inputs - 1):
                next_warm_up = input_batch[t_hat + 1, :, :]
                warm_up_input = torch.cat((warm_up_input, next_warm_up), 1)
            warm_up_list.append(self.warm_up_network_list[l](warm_up_input))
        return torch.stack(warm_up_list)

    def __warm_up_lstm(self, input_batch):
        warm_up_c_list = []
        warm_up_h_list = []
        for l in range(self.num_layers):
            warm_up_input = input_batch[0, :, :]
            for t_hat in range(self.warm_up_inputs - 1):
                next_warm_up = input_batch[t_hat + 1, :, :]
                warm_up_input = torch.cat((warm_up_input, next_warm_up), 1)
            warm_up_hc_l = self.warm_up_network_list[l](warm_up_input)
            warm_up_h_l = warm_up_hc_l[:, :self.latent_dim]
            warm_up_c_l = warm_up_hc_l[:, self.latent_dim:]
            warm_up_h_list.append(warm_up_h_l)
            warm_up_c_list.append(warm_up_c_l)
        warm_up_h = torch.stack(warm_up_h_list)
        warm_up_c = torch.stack(warm_up_c_list)
        return warm_up_h, warm_up_c

    def forward_n_step(self, obs_batch, train_schedule, predict_deltas, factor_delta, action_batch=None):

        forward_for_planning = action_batch is not None

        if forward_for_planning:
            action_dim = action_batch.size()[2]
        seq_len, batch_size, obs_dim = obs_batch.size()

        # Scheduled sampling
        sampling_rand = np.random.rand(seq_len, batch_size)
        sampling_schedule_np = np.clip(np.ceil(train_schedule - sampling_rand), 0, 1)
        sampling_schedule = torch.unsqueeze(torch.from_numpy(sampling_schedule_np), dim=2).expand(seq_len, batch_size,
                                                                                                  obs_dim).float()
        cs = []
        hs = []
        outs = []
        deltas = []
        last_hc = None
        last_out = None

        for t in range(seq_len):

            if last_out is None:
                x_t = obs_batch[t, :, :]
            else:
                x_t = sampling_schedule[t, :, :] * obs_batch[t, :, :] + (1 - sampling_schedule[t, :, :]) * last_out[0, :, :]


            x_t = x_t.expand([1, batch_size, obs_dim])

            if forward_for_planning:
                a_t = action_batch[t, :, :].expand([1, batch_size, action_dim])
                x_t_full = torch.cat((a_t, x_t), 2)
            else:
                x_t_full = x_t


            if t == 0:

                # Latent state initialization
                if self.use_warm_up:
                    if forward_for_planning:
                        input_action_batch = torch.cat((action_batch, obs_batch), 2)
                    else:
                        input_action_batch = obs_batch

                    if self.rnn_type == 'LSTM':
                        last_hc = self.__warm_up_lstm(input_action_batch)
                    else:
                        last_hc = self.__warm_up(input_action_batch)

            y, h, c = self.__forward(x_t_full, last_hc)

            if self.rnn_type == 'LSTM':
                # For LSTMs the hidden state is a tuple
                cs.append(c)
                last_hc = (h, c)
            else:
                last_hc = h

            last_out = y
            if predict_deltas:
                deltas.append(torch.squeeze(y))
                last_out = x_t + factor_delta * y
            hs.append(h)
            outs.append(torch.squeeze(last_out))

        hs_final = torch.stack(hs)
        out_final = torch.stack(outs)
        cs_final = hs_final

        delta_final = None
        if predict_deltas:
            delta_final = torch.stack(deltas)

        # For cell states and latent states we move the layer number to the first dimension to match GateL0RD
        if self.rnn_type == 'LSTM':
            cs_final = torch.stack(cs)

        # 5 outputs to match GateL0RD but no gate activation included
        return out_final, cs_final.permute([1, 0, 2, 3]), None, hs_final.permute([1, 0, 2, 3]), delta_final