import torch

def sample_neighbors(X, num_samples=5, threshold=1e-3):
    u, s, vh = torch.linalg.svd(X, full_matrices=False)
    x_var_exp = torch.cumsum(s ** 2, dim=-1) / torch.sum(s ** 2, dim=-1).unsqueeze(-1)

    idx = torch.where(x_var_exp > 1 - threshold)[1][0] + 1
    s_len = s.shape[-1]

    s_eye = torch.eye(s_len, device=X.device)
    s_eye_rep = torch.tile(s_eye, (num_samples, X.shape[0], 1, 1))

    diag = torch.rand(size=([num_samples] + list(s.shape)), device=X.device)
    diag[..., :idx] = 1

    s_tile = torch.tile(s, (num_samples, 1, 1))
    u_tiled = torch.tile(u, (num_samples, 1, 1, 1))
    vh_tiled = torch.tile(vh, (num_samples, 1, 1, 1))

    s_tile_scaled = s_tile * diag
    diag_tile = s_tile_scaled[..., None] * s_eye_rep

    res = u_tiled @ diag_tile @ vh_tiled
    return res
def get_settings(args, ii=0):
    setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_id,
        args.model,
        args.mode_select,
        args.modes,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        ii)
    return setting
