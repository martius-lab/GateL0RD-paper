import torch

def calc_gate_rate(gs):
    seq_len, batch_size, dim = gs.shape
    overall = seq_len * batch_size * dim
    gate_open_indices = torch.nonzero(gs[:, :, :], as_tuple=True)
    return float(len(gs[gate_open_indices]))/overall

def calc_gate_open_times(gs):
    seq_len, batch_size, _ = gs.shape
    gs_summed = torch.sum(gs, 2)
    overall = seq_len * batch_size
    gate_open_indices = torch.nonzero(gs_summed[:, :], as_tuple=True)
    return float(len(gs_summed[gate_open_indices]))/overall

def calc_gate_dims_used(gs):
    seq_len, batch_size, dim = gs.shape
    gs_summed = torch.sum(gs, 0)
    overall = dim * batch_size
    gate_open_indices = torch.nonzero(gs_summed[:, :], as_tuple=True)
    return float(len(gs_summed[gate_open_indices]))/overall