"""Piecewise smoothing scheme for labels."""

import numpy as np
import torch


class SmoothLDL:
    """Implementation of smoothing scheme."""

    def __init__(self, eps_min=0.1, eps_max=1.0, sigma=3.0, eps_type="piecewise"):
        """Init the object."""
        self.acne_axis = torch.arange(0, 65, 1)
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_type = eps_type
        self.sigma = sigma

    def smooth_labels(self, y_true):
        """Perform label smoothing."""
        y_true = y_true[:, None]
        # get target in one-hot format
        y_oh = torch.nn.functional.one_hot(y_true, num_classes=65)[:, -1]
        # get epsilon function that depends on lesion counting number
        eps_cou = SmoothLDL.get_smooth_param(
            eps_min=self.eps_min,
            eps_max=self.eps_max,
            eps_type=self.eps_type,
        )
        # eps for every image in a batch
        eps_batch = eps_cou[y_true]

        self.sigma = torch.tensor(self.sigma)
        pi = torch.tensor(3.141592653)
        noise = (
            1
            / torch.sqrt(2 * pi * self.sigma**2)
            * torch.exp(-((y_true - self.acne_axis) ** 2) / (2 * self.sigma**2))
        )
        noise /= torch.sum(noise, axis=1)[:, None]

        # perform smoothing
        y_smooth = (1 - eps_batch) * y_oh + eps_batch * noise

        return y_smooth.float().cuda()

    def get_smooth_param(eps_min, eps_max, eps_type):
        """Evaluate epsilon map w.r.t. number of acne on photo."""
        if eps_type == "piecewise":

            def eps_line(x, k, b):
                return k * x + b

            eps_1 = eps_line(np.arange(0, 5, 1), (eps_max - eps_min) / 4, eps_min)  # severity 0
            eps_2 = eps_line(np.arange(0, 8, 1), -(eps_1[-1] - eps_min) / 7, eps_1[-1])  # severity 1
            eps_3 = eps_line(np.arange(1, 8, 1), (eps_2[0] - eps_min) / 7, eps_2[-1])  # severity 1
            eps_4 = eps_line(np.arange(0, 15, 1), -(eps_3[-1] - eps_min) / 14, eps_3[-1])  # severity 2
            eps_5 = eps_line(np.arange(0, 15, 1), (eps_4[0] - eps_min) / 14, eps_4[-1])  # severity 2
            eps_6 = eps_line(np.arange(0, 15, 1), -(eps_5[-1] - eps_min) / 14, eps_5[-1])  # severity 3

            return torch.tensor(np.concatenate((eps_1, eps_2, eps_3, eps_4, eps_5, eps_6)))
