import torch.nn as nn

'''
Helpers to create 
- f_pre, preprocessing MLP of inputs
- f_post, read out layers of network output
- f_init, "warm up" layers setting the initial latent state
'''

def create_f_pre(f_pre_layers, input_dim, feature_dim):
    module_pre = nn.ModuleList([])
    h_dim = input_dim
    for pre_l in range(f_pre_layers):
        # Fan in type of network, decreasing features per layer
        pre_l_factor = pow(2, (f_pre_layers - pre_l - 1))
        module_pre.append(nn.Linear(h_dim, pre_l_factor * feature_dim))
        module_pre.append(nn.Tanh())
        h_dim = pre_l_factor * feature_dim
    return nn.Sequential(*module_pre)


def create_f_init(f_init_layers, f_init_inputs, input_dim, feature_dim, latent_dim):
    input_dim_warm_up = input_dim * f_init_inputs
    feature_dim_warm_up = feature_dim
    warm_up_net = nn.ModuleList([])
    for w in range(f_init_layers):
        w_factor = pow(2, (f_init_layers - w - 1))
        if w == (f_init_layers - 1):
            feature_dim_warm_up = latent_dim
        warm_up_net.append(nn.Linear(input_dim_warm_up, w_factor * feature_dim_warm_up))
        warm_up_net.append(nn.Tanh())
        input_dim_warm_up = w_factor * feature_dim
    return nn.Sequential(*warm_up_net)


def create_f_post(f_post_layers, feature_dim, output_dim):
    post_module = nn.ModuleList([])
    in_post = feature_dim
    h_dim = feature_dim
    for post_l in range(f_post_layers):
        h_factor = pow(2, (f_post_layers - post_l - 2))
        if post_l == f_post_layers - 1:
            h_factor = 1
            h_dim = output_dim
        post_module.append(nn.Linear(in_post, h_factor * h_dim))
        if post_l < (f_post_layers - 1):
            post_module.append(nn.Tanh())
        in_post = h_factor * h_dim
    return nn.Sequential(*post_module)