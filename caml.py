import torch
import numpy as np
from intrinsic_dimension import intrinsic_dimension
from sklearn.neighbors import NearestNeighbors


def torch_knn(X, K):
    """
    Compute K-nearest neighbors using PyTorch.

    Args:
        X (torch.Tensor or ndarray): Input data points.
        K (int): Number of nearest neighbors to consider.

    Returns:
        tuple: Distances and indices of the K-nearest neighbors.
    """
    if not torch.is_tensor(X) or X.is_cuda:
        X = torch.tensor(X, device='cuda', dtype=torch.float32)

    c_dist = torch.cdist(X, X, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    t_k = torch.topk(c_dist, K + 1, dim=0, largest=False, sorted=True)

    return t_k.values.cpu().numpy().T, t_k.indices.cpu().numpy().T


def caml(X, K=15, d=None, dist_matrix=None, XK=None, use_gpu=True, batch_size=128, verbose=False):
    """
    Estimates the principal curvatures of a system of points X using curvature-aware manifold learning.

    Args:
        X (ndarray): Data matrix (n, D) where n is the number of points and D is the extrinsic dimension.
        K (int): Number of nearest neighbors to consider.
        d (int): Intrinsic dimension of the data. If None, it is estimated.
        dist_matrix (ndarray): Precomputed distance matrix.
        XK (ndarray): Precomputed nearest neighbors.
        use_gpu (bool): Whether to use GPU for computations.
        batch_size (int): Batch size for estimation.
        verbose (bool): If True, print progress information.

    Returns:
        ndarray: Principal curvatures of the data.
    """
    # Step 1: Find K-nearest neighbors for each point
    XK = _get_neighbors(X, K, dist_matrix, XK, use_gpu)

    # Step 2: Estimate intrinsic dimension if not provided
    if d is None:
        d = intrinsic_dimension(X, XK, use_gpu, batch_size, verbose)

    # Step 3: Compute curvature
    if use_gpu:
        return _caml_batched_gpu(X, XK, d, batch_size, verbose).detach().cpu().numpy()
    else:
        return _caml_batched_cpu(X, XK, d, batch_size, verbose)


def _get_neighbors(X, K, dist_matrix, XK, use_gpu):
    """
    Helper function to get K-nearest neighbors.

    Args:
        X (ndarray): Input data points.
        K (int): Number of neighbors.
        dist_matrix (ndarray): Precomputed distance matrix.
        XK (ndarray): Precomputed nearest neighbors.
        use_gpu (bool): Whether to use GPU for computations.

    Returns:
        ndarray: Nearest neighbors of the data points.
    """
    if XK is None:
        if dist_matrix is None:
            if use_gpu:
                I = torch_knn(X, K)[1]
            else:
                nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm='brute', metric='minkowski')
                nbrs.fit(X)
                I = nbrs.kneighbors(X, return_distance=False)[:, 1:]
            XK = X[I]
        else:
            nbrs = NearestNeighbors(n_neighbors=K + 1, metric='precomputed')
            nbrs.fit(dist_matrix)
            I = nbrs.kneighbors(dist_matrix, return_distance=False)[:, 1:]
            XK = X[I]
    return XK


def _caml_batched_cpu(X, XK, d, batch_size, verbose):
    """
    CPU-based batched computation of curvature-aware manifold learning.

    Args:
        X (ndarray): Input data points.
        XK (ndarray): Nearest neighbors.
        d (int): Intrinsic dimension.
        batch_size (int): Batch size.
        verbose (bool): Verbosity flag.

    Returns:
        ndarray: Estimated principal curvatures.
    """
    # Initialize structures for storing curvature values
    num_samples = X.shape[0]
    p_curvs = []

    # Loop through data in batches
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        if verbose:
            print(f"Processing batch {i} to {end_idx}")

        batch_X = X[i:end_idx]
        batch_XK = XK[i:end_idx]

        p_curvs.append(_caml_inner_cpu(batch_X, batch_XK, d))

    return np.concatenate(p_curvs, axis=0)



def _caml_inner_cpu(X_numpy, XK_numpy, d):
    """
    NumPy implementation of inner computation for curvature estimation.

    Args:
        X_numpy (np.ndarray): Input points in NumPy.
        XK_numpy (np.ndarray): Nearest neighbors in NumPy.
        d (int): Intrinsic dimension.

    Returns:
        np.ndarray: Curvature information for the given points.
    """
    tidx_numpy = np.triu_indices(d, d)
    ones_mult_numpy = np.ones((d, d))
    np.fill_diagonal(ones_mult_numpy, 0.5)

    XK_numpy = np.transpose(XK_numpy, (0, 2, 1))
    X_numpy = np.expand_dims(X_numpy, -1)

    X_b_numpy = XK_numpy - X_numpy
    U, S, Vt = np.linalg.svd(X_b_numpy - np.mean(X_b_numpy, axis=-1, keepdims=True), full_matrices=False)
    Ui_b_numpy = np.matmul(np.transpose(U, (0, 2, 1)), X_b_numpy)

    Ui_d_b_numpy = Ui_b_numpy[:, :d]
    Ui_d_tr_b_numpy = np.transpose(Ui_d_b_numpy, (0, 2, 1))
    fi_b_numpy = np.transpose(Ui_b_numpy[:, d:], (0, 2, 1))

    UUi_b_numpy = np.einsum('bki,bkj->bkij', Ui_d_tr_b_numpy, Ui_d_tr_b_numpy)
    UUi_b_numpy = UUi_b_numpy * ones_mult_numpy
    UUi_b_numpy = UUi_b_numpy[:, :, tidx_numpy[0], tidx_numpy[1]].transpose(0, 2, 1)

    psii_b_numpy = np.concatenate((Ui_d_b_numpy, UUi_b_numpy), axis=1).transpose(0, 2, 1)
    Bi_b_numpy, _, _, _ = np.linalg.lstsq(psii_b_numpy, fi_b_numpy, rcond=None)
    Bi_b_numpy = np.transpose(Bi_b_numpy, (0, 2, 1))

    Hi_b_numpy = np.zeros((XK_numpy.shape[0], Bi_b_numpy.shape[1], d, d), dtype=Bi_b_numpy.dtype)
    Hi_b_numpy[:, :, tidx_numpy[0], tidx_numpy[1]] = Bi_b_numpy[:, :, d:]
    Hi_b_numpy[:, :, tidx_numpy[1], tidx_numpy[0]] = Bi_b_numpy[:, :, d:]

    eig_values = np.linalg.eigvalsh(Hi_b_numpy)
    R_b_numpy = eig_values.reshape((eig_values.shape[0], -1))

    return R_b_numpy


def _caml_batched_gpu(X, XK, d, batch_size, verbose):
    """
    GPU-based batched computation of curvature-aware manifold learning.

    Args:
        X (torch.Tensor): Input data points.
        XK (torch.Tensor): Nearest neighbors.
        d (int): Intrinsic dimension.
        batch_size (int): Batch size.
        verbose (bool): Verbosity flag.

    Returns:
        torch.Tensor: Estimated principal curvatures.
    """
    # Convert data to GPU tensors
    X = torch.tensor(X, device='cuda', dtype=torch.float32)
    XK = torch.tensor(XK, device='cuda', dtype=torch.float32)

    # Initialize for storing curvature values
    num_samples = X.shape[0]
    p_curvs = []

    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        if verbose:
            print(f"Processing batch {i} to {end_idx}")

        batch_X = X[i:end_idx]
        batch_XK = XK[i:end_idx]

        p_curvs.append(_caml_inner_gpu(batch_X, batch_XK, d))

    return torch.cat(p_curvs, dim=0)


def _caml_inner_gpu(X_torch, XK_torch, d):
    """
    GPU implementation of inner computation for curvature estimation.

    Args:
        X_torch (torch.Tensor): Input points on GPU.
        XK_torch (torch.Tensor): Nearest neighbors on GPU.
        d (int): Intrinsic dimension.

    Returns:
        torch.Tensor: Curvature information for the given points.
    """
    tidx_torch = torch.triu_indices(d, d)
    ones_mult_torch = torch.ones((d, d), device='cuda')
    ones_mult_torch.fill_diagonal_(0.5)

    XK_torch = XK_torch.transpose(2, 1)
    X_torch = X_torch.unsqueeze(-1)

    X_b_torch = XK_torch - X_torch
    DUi_b_torch, _, _ = torch.linalg.svd(X_b_torch, full_matrices=False)
    Ui_b_torch = DUi_b_torch.transpose(2, 1) @ X_b_torch

    Ui_d_b_torch = Ui_b_torch[:, :d]
    Ui_d_tr_b_torch = Ui_d_b_torch.transpose(-2, -1)
    fi_b_torch = Ui_b_torch[:, d:].transpose(-2, -1)

    UUi_b_torch = torch.einsum('bki,bkj->bkij', Ui_d_tr_b_torch, Ui_d_tr_b_torch)
    UUi_b_torch = UUi_b_torch * ones_mult_torch
    UUi_b_torch = UUi_b_torch[:, :, tidx_torch[0], tidx_torch[1]].transpose(-2, -1)

    psii_b_torch = torch.cat((Ui_d_b_torch, UUi_b_torch), dim=1).transpose(-2, -1)
    Bi_b_torch = torch.linalg.lstsq(psii_b_torch, fi_b_torch).solution.transpose(-2, -1)

    Hi_b_torch = torch.zeros((XK_torch.shape[0], Bi_b_torch.shape[1], d, d), dtype=Bi_b_torch.dtype, device='cuda')
    Hi_b_torch[:, :, tidx_torch[0], tidx_torch[1]] = Bi_b_torch[:, :, d:]
    Hi_b_torch[:, :, tidx_torch[1], tidx_torch[0]] = Bi_b_torch[:, :, d:]

    eig_values = torch.linalg.eigvalsh(Hi_b_torch)
    R_b_torch = eig_values.reshape((eig_values.shape[0], -1))

    return R_b_torch
