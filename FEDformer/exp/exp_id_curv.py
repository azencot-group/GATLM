import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from FEDformer.data_provider.data_factory import data_provider
from FEDformer.exp.exp_basic import Exp_Basic
from FEDformer.models import FEDformer, Autoformer, Informer, Transformer
from os.path import join
from intrinsic_dimension import intrinsic_dimension
from utils import sample_neighbors
from caml import caml
warnings.filterwarnings('ignore')


class Exp_id_curv(Exp_Basic):
    def __init__(self, args):
        super(Exp_id_curv, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def estimate_ids(self, data, verbose=False):
        """
        Estimate IDs for each layer in the data using a given estimator.

        Args:
            data (list of np.ndarray): List of data layers to process.

        Returns:
            list: List of estimated IDs for each layer (if successful).
        """

        ids = []
        for idx, layer in enumerate(data):
            id_est = intrinsic_dimension(layer,verbose=verbose)
            ids.append(id_est)
            if verbose:
                print(f'Layer {idx}: ID estimate = {id_est}')
        return ids

    def save_ids(self, args, settings):
        id_dir = join(args.save_path, 'ids')
        id_path = join(id_dir, f'ids_{args.task_id}_{args.model}.npy')

        if not os.path.exists(id_dir):
            os.makedirs(id_dir)

        latent_rep = self.get_latent_rep(args, settings)
        ids = self.estimate_ids(latent_rep)
        np.save(id_path, ids)

    def save_curvs(self, args, settings):
        id_dir = join(args.save_path, 'ids')
        id_path = join(id_dir, f'ids_{args.task_id}_{args.model}.npy')
        curv_dir = join(args.save_path, 'curvs')
        curvs_path = join(curv_dir, f'curvs_{args.task_id}_{args.model}.npy')

        if not os.path.exists(id_path):
            self.save_ids(args, settings)
        ids = np.load(id_path)
        ids = np.ceil(ids).astype(int)
        if not os.path.exists(curv_dir):
            os.makedirs(curv_dir)

        # curvs = self.est_curvature(settings, ids)
        curvs = self.est_curvs1(settings, ids)
        np.save(curvs_path, curvs)

    def get_latent_rep(self, args, setting):
        """
        Estimate the intrinsic dimensions of the latent data using the model predictions.

        Args:
            setting: Model checkpoint setting.

        Returns:
            latent_data: Latent representations as a list of numpy arrays.
        """
        _, loader = self._get_data(flag='train')
        print('Loading model...')

        # Load model
        self.model.load_state_dict(torch.load(os.path.join(args.checkpoints + setting, 'checkpoint.pth')))
        self.model.eval()

        num_samples = args.id_samples
        sample_size = args.batch_size * args.seq_len
        act_num_batches = int(np.ceil(num_samples / sample_size)) * args.batch_size
        arr_size = min(act_num_batches, len(loader) * args.batch_size)
        latent_data = None

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                if i * sample_size >= num_samples:
                    break

                print(f"Processing batch {i}")

                # Move data to device
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._move_to_device(batch_x, batch_y, batch_x_mark,
                                                                                    batch_y_mark)

                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(self.device)

                # Forward pass through model
                _, latent_data_i = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Initialize latent data storage
                if latent_data is None:
                    latent_data = [np.empty(((arr_size,) + latent_data_i[j].shape[1:])) for j in
                                   range(len(latent_data_i))]

                # Store latent data
                for jj in range(len(latent_data)):
                    latent_data[jj][i * args.batch_size:(i + 1) * args.batch_size] = latent_data_i[jj]

            # Reshape latent data
            latent_data = [latent_data[i].reshape(-1, latent_data[i].shape[-1]) for i in range(len(latent_data))]

        return latent_data

    def est_curv(self, data, data_neigh, ids, batch_size=1024):
        """
        Estimate curvature for each layer using CAML.

        Args:
            data (list): List of data layers.
            data_neigh (list): List of neighboring data layers.
            ids (list): List of IDs corresponding to each layer.
            batch_size (int, optional): Batch size for processing. Defaults to 2^10.

        Returns:
            list: List of curvature estimates for each layer.
        """
        curvs = []

        for jj, (layer, layer_neigh) in enumerate(zip(data, data_neigh)):
            # Estimate curvature using CAML for each layer
            crv_all = caml(X=layer, d=ids[jj], XK=layer_neigh, batch_size=batch_size, use_gpu=True)
            curv_mean = np.mean(np.abs(crv_all), axis=1)

            curvs.append(curv_mean)

        return curvs

    def est_curvature(self, setting, ids, num_neigh=2 ** 6):
        """
        Estimate the curvature of the latent data using model predictions and neighbor generation.

        Args:
            setting: Model checkpoint setting.
            ids: IDs for curvature calculation.
            curv_path: Optional path for saving curvature results.
            num_neigh: Number of neighbors for curvature estimation.
            ii: Iteration index (not currently used).

        Returns:
            curvs: Estimated curvatures as a numpy array.
        """
        # Load data and model
        _, loader = self._get_data(flag='train')
        print('Loading model...')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))
        self.model.eval()

        num_samples = self.args.curv_samples
        sample_size = self.args.batch_size
        curvs = []

        with torch.no_grad():
            # Iterate through test_loader to compute latent data and curvature
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                if i * sample_size >= num_samples:
                    break

                print(f"Processing batch {i}")

                # Move data to device
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._move_to_device(batch_x, batch_y, batch_x_mark,
                                                                                    batch_y_mark)

                # Generate latent data from model
                latent_data_i = self._generate_latent_data(batch_x, batch_x_mark, batch_y, batch_y_mark)

                # Generate neighbors for batch_x and process through model
                batch_x_neigh = sample_neighbors(batch_x, num_neigh)
                batch_x_neigh_latent = self._generate_latent_neighbors(batch_x_neigh, batch_x_mark, batch_y,
                                                                       batch_y_mark)

                # Estimate curvature
                try:
                    curvs_i = self.est_curv(latent_data_i, batch_x_neigh_latent, ids)
                    curvs.append(curvs_i)
                except Exception as e:
                    print(f'Iteration failed: {e}')
                    continue

        # Concatenate curvature estimates
        curvs = self._concatenate_curvatures(curvs)

        return curvs

    def _move_to_device(self, *tensors):
        """
        Move tensors to the specified device.

        Args:
            tensors: Tensors to be moved.

        Returns:
            List of tensors moved to the appropriate device.
        """
        return [tensor.float().to(self.device) for tensor in tensors]

    def _generate_latent_data(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        """
        Generate latent data using the model based on the input data.

        Args:
            batch_x: Input features.
            batch_x_mark: Input feature marks.
            batch_y: Target features.
            batch_y_mark: Target feature marks.

        Returns:
            latent_data_i: Latent representation of the data.
        """
        # Prepare decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

        # Forward pass through model
        _, latent_data_i = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        latent_data_i = [latent_data_i[i].reshape(latent_data_i[i].shape[0], -1) for i in range(len(latent_data_i))]

        return latent_data_i

    def _generate_latent_neighbors(self, batch_x_neigh, batch_x_mark, batch_y, batch_y_mark):
        """
        Generate latent representations for neighboring samples.

        Args:
            batch_x_neigh: Neighboring samples generated from batch_x.
            batch_x_mark: Input feature marks.
            batch_y: Target features.
            batch_y_mark: Target feature marks.

        Returns:
            batch_x_neigh_latent: Latent representations of neighboring samples.
        """
        batch_x_neigh_latent = []
        dec_inp_gen = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()

        for inputs_gen_i in batch_x_neigh:
            dec_inp_gen_i = torch.cat([inputs_gen_i[:, -self.args.label_len:, :], dec_inp_gen], dim=1).to(self.device)
            _, latent_rep_i = self.model(inputs_gen_i, batch_x_mark, dec_inp_gen_i, batch_y_mark)
            latent_rep_i = [latent_rep_i[i].reshape(latent_rep_i[i].shape[0], -1) for i in range(len(latent_rep_i))]
            batch_x_neigh_latent.append(latent_rep_i)

        batch_x_neigh_latent = [
            np.stack([batch_x_neigh_latent[j][i] for j in range(len(batch_x_neigh_latent))], axis=1)
            for i in range(len(batch_x_neigh_latent[0]))
        ]

        return batch_x_neigh_latent

    def _concatenate_curvatures(self, curvs):
        """
        Concatenate curvature estimates across batches.

        Args:
            curvs: List of curvature estimates from different batches.

        Returns:
            Concatenated curvature estimates as a numpy array.
        """
        return np.array([
            np.concatenate([curvs[j][i] for j in range(len(curvs))], axis=0) for i in range(len(curvs[0]))
        ], dtype=object)

