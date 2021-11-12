import torch
import torch.nn as nn
import models.gates as gates


class GateL0RDCell(nn.Module):

    def __init__(self,  input_dim, output_dim, latent_dim, num_layers_internal=1,
                 stochastic_gates=True, gate_noise_level=0.1, concat_gates=True,
                 gate_type='ReTanh', output_gate=True, device=None):
        
        super(GateL0RDCell, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim


        input_dim_gates = input_dim + latent_dim

        # Create g-network:
        temp_gating = nn.ModuleList([])
        in_dim_g = input_dim_gates
        for gl in range(num_layers_internal):
            gl_factor = pow(2, (num_layers_internal - gl - 1))
            out_dim_g = gl_factor * latent_dim
            temp_gating.append(nn.Linear(in_dim_g, out_dim_g))

            if gl < (num_layers_internal - 1):
                temp_gating.append(nn.Tanh())
            in_dim_g = out_dim_g
        self.input_gates = nn.Sequential(*temp_gating)

        # Create r-network:
        temp_r_function = nn.ModuleList([])
        in_dim_r = input_dim_gates
        for rl in range(num_layers_internal):
            rl_factor = pow(2, (num_layers_internal - rl - 1))
            out_dim_r = rl_factor * latent_dim
            temp_r_function.append(nn.Linear(in_dim_r, out_dim_r))
            temp_r_function.append(nn.Tanh())
            in_dim_r = out_dim_r
        self.r_function = nn.Sequential(*temp_r_function)

        # Create output function p * o:

        # Create p-network:
        temp_output_function = nn.ModuleList([])
        temp_output_function.append(nn.Linear(input_dim_gates, output_dim))
        temp_output_function.append(nn.Tanh())
        self.output_function = nn.Sequential(*temp_output_function)

        # Create o-network
        if output_gate:
            temp_outputgate = nn.ModuleList([])
            temp_outputgate.append(nn.Linear(input_dim_gates, output_dim))
            temp_outputgate.append(nn.Sigmoid())
            self.output_gates = nn.Sequential(*temp_outputgate)

        # Create the gate function and the regularization function
        if gate_type == 'ReTanh' or gate_type == 'Step':

            if gate_type == 'ReTanh':
                self.gate = gates.ReTanh.apply
            else:
                self.gate = gates.HeavisideST.apply

            if concat_gates:
                self.gate_reg = gates.HeavisideST.apply
            else:
                self.gate_reg = gates.HeavisideReLU.apply
        elif gate_type == 'Sigmoid':
            self.gate = gates.Sigmoid.apply
            self.gate_reg = gates.Sigmoid.apply
        else:
            assert False, 'Gate type ' + str(gate_type) + ' unknown.'


        self.output_gate = output_gate
        self.concat_gates = concat_gates
        self.stochastic_gates = stochastic_gates
        self.gate_noise_level = gate_noise_level

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device


    def forward(self, x_t, h_tminus1=None):
        """
        Forward pass one step, i.e. pass through g-, r-, p- and o-subnetwork
        :param x_t: tensor of cell inputs
        :param h_tminus1: tensor of last hidden state
        :return: rnn output, hidden states, tensor of gatings
        """
        assert len(x_t.shape) ==2 and len(h_tminus1.shape) == 2, "Wrong input dimensionality in GateL0RDCell: " + str(x_t.shape) + " and " + str(h_tminus1.shape)
        batch_size, layer_input_dim = x_t.size()

        if h_tminus1 is None:
            h_tminus1 = torch.zeros((batch_size, self.latent_dim), device=self.device)
        else:
            assert h_tminus1.shape[1] == self.latent_dim

        # Layer input to g and r-network is the current input plus the last cell state
        gr_layer_input = torch.cat((x_t, h_tminus1), 1)

        '''
        G- NETWORK
        '''
        i_t = self.input_gates(gr_layer_input)
        if self.stochastic_gates and self.training:
            gate_noise = torch.randn(size=(batch_size, self.latent_dim), device=self.device) * self.gate_noise_level
        else:
            # Gate noise is zero
            factor_gate_noise = 0.0
            gate_noise = torch.ones((batch_size, self.latent_dim),  device=self.device) * factor_gate_noise

        # Stochastic input gate activation
        Lambda_t = self.gate(i_t - gate_noise)
        if not self.concat_gates:
            Theta_t = self.gate_reg(i_t - gate_noise)
        else:
            Theta_t = self.gate_reg(Lambda_t)

        '''
        R-Network
        '''
        h_hat_t = self.r_function(gr_layer_input)

        '''
        New latent state
        '''
        h_t = Lambda_t * h_hat_t + (1.0 - Lambda_t) * h_tminus1

        '''
        Output function :
        '''
        xh_t = torch.cat((x_t, h_t), 1)

        y_hat_t = self.output_function(xh_t)
        if self.output_gate:
            # Output is computed as p(x_t, h_t) * o(x_t, h_t)
            o_lt = self.output_gates(xh_t)
            y_t = y_hat_t * o_lt
        else:
            # Output is computed p(x_t, h_t)
            y_t = y_hat_t

        return y_t, h_t, Theta_t


    def loss(self, loss_task, Theta):
        gate_loss = torch.mean(Theta)
        return loss_task + self.reg_lambda * gate_loss

