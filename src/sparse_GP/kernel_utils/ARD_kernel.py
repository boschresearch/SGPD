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

import itertools
import math
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch.autograd import Function, Variable, grad

FloatTensor = torch.FloatTensor
Parameter = torch.nn.Parameter

def full_kernel(left_var: torch.Tensor, right_var: torch.Tensor, sigma_f: torch.Tensor, L_sqrt: torch.Tensor) -> torch.Tensor:
    """ computes the kernel function of two inputs K with K_ij = k(x_i, x_j)
    Args:
        left_var ((batchsize, dim) - torch.Tensor): left variables.
        right_var ((batchsize, dim) - torch.Tensor): right variable.

    Returns:
        K (TYPE): DESCRIPTION.

    """
    # divide by lengthscale
    X = left_var / L_sqrt
    X2 = right_var / L_sqrt 
    Xs = (X ** 2).sum(dim=1)
    X2s = (X2 ** 2).sum(dim=1)
    dist = -2 * X.mm(X2.t()) + Xs.view(-1, 1) + X2s.view(1, -1)
    K = sigma_f*torch.exp(-0.5 * dist)
    return K