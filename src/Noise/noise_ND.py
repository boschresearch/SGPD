# -*- coding: utf-8 -*-
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Katharina Ensinger, katharina.ensinger@de.bosch.com

"""probability computation with respect to noise level"""

import numpy as np
import torch
from torch.distributions import Normal


class GaussianNoise(torch.nn.Module):
    """initialization of noise and computation of \sum_n log(p(\hat{x}_n|x_n))

    Args:
        d (int): dimension of problem.
        noiselevel (torch.Tensor): initialization of expected noise
        device (torch.device): device

    Attributes:
        d (int, optional): dimension of problem. Defaults to 2.
        noiselevel (torch.Tensor): initialization of expected noise
        device (torch.device): device
    """

    def __init__(self, d: int, noiselevel: torch.Tensor, device: torch.device = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()
        self.n_x = torch.nn.Parameter(torch.log(noiselevel * torch.ones(1, d, dtype=torch.double)), requires_grad=True)
        self.dim = d
        self.to(self.device)

    def forward(self, x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
        """ computes \sum_n log(p(\hat{x}_n|x_n))

        Args:
            x (torch.Tensor): (batchsize, dim)- input.
            xhat (torch.Tensor): (batchsize,dim)- noisy observations.

        Returns:
            torch.Tensor: log-probability.

        """
        p_func = Normal(x, torch.exp(self.n_x))
        dist = p_func.log_prob(xhat)
        return torch.sum(dist)
