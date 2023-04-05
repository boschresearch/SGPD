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

"""symplectic midpoint method"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from torch.autograd import Function, Variable, grad

sys.path.append("../")
import itertools
import math

import scipy 

from integrator.Implicit.IFT import AbstractProblem, Argmin
from integrator.Implicit.Utils import clone_state_dict

FloatTensor = torch.FloatTensor
Parameter = torch.nn.Parameter
import random
from typing import Optional

from torch.distributions import Normal

dt = 0.1


class ImplicitMidpoint(AbstractProblem):
    """implicit integrator method provides solve and custom backward.

    Args:
        ode (torch.nn.Module): dynamics.

    Attributes:
        method (str): specify type of implicit integrator
        ode (torch.nn.Module): dynamics.
        _rtol (float): tolerance for inner optimization problem
        _lr (float): learning rate for inner optimization problem
        _max_iter (flaot): maximum number of training steps
    """

    def __init__(
        self, ode,
    ):
        super().__init__()
        self.method = "midpoint"
        self.ode = ode
        self._rtol = 1e-8
        self._lr = 1e-3
        self._max_iter = 1000
        
    def forward(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            u (torch.Tensor): (1, dim)-solution of self.solve.
            x (torch.Tensor): (1,dim)- input of integrator step.

        Returns:
            evaluation of minimization problem for backward pass

        """

        J = torch.tensor([[0, -1.0], [1.0, 0]], dtype=torch.float)
        r2 = (u - (x + (torch.matmul(J, (self.ode((x + u) / 2)).t() * dt)).view(1,2)[0])) ** 2

        return r2.sum()


    def solve(self, x: torch.Tensor) -> torch.Tensor:
        """Solution of problem needs to be a flattened vector"""
        J = torch.tensor([[0, -1.0], [1.0, 0]], dtype=torch.float)
        def scipy_ode(x_next):
            nextStep = x.view(1, 2).detach().numpy() + (torch.matmul(J, (self.ode((x + torch.from_numpy(x_next).float()) / 2).float()).t()) * dt).view(1, 2).detach().numpy()
            r2 = x_next - nextStep
            return r2.reshape(2)

        x_next = (x + dt * self.ode(x)[0]).detach().numpy()
        optResult = scipy.optimize.least_squares(scipy_ode, x_next.reshape(2), method="lm")
        x_best = optResult.x
        return torch.from_numpy(x_best).view(1, 2).float()