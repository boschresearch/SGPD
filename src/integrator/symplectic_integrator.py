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

"""explicit symplectic integrators"""

import itertools
import math
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as inter
import torch
from scipy.integrate import odeint
from torch.autograd import Function, Variable, grad

FloatTensor = torch.FloatTensor
Parameter = torch.nn.Parameter

random.seed(30)
np.random.seed(30)
torch.manual_seed(30)


class SymInt(torch.nn.Module):
    """symplectic integator for torch.nn.module

    Args:
        d (int, optional): dimension of problem. Defaults to 2.
        method (str, optional): specification of integrator. Defaults to "explicitEuler".
        p_dot (torch.nn.module, optional): dynamics velocity
        q_dot (torch.nn.module, optional): dynamics position

    Attributes:
        d (int, optional): dimension of problem. Defaults to 2.
        method (str, optional): specification of integrator. Defaults to "explicitEuler".
        p_dot (torch.nn.module, optional): dynamics velocity
        q_dot (torch.nn.module, optional): dynamics position
        dNew (int): dimension to cut into two halfes
    """

    def __init__(
        self,
        d: int,
        method: Optional[str] = None,
        p_dot: Optional[torch.nn.Module] = None,
        q_dot: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.d = d
        self.p_dot = p_dot
        self.q_dot = q_dot
        self.dNew = int(self.d / 2)
        self.method = method

    def forward(self, dt: float, x: torch.Tensor) -> torch.Tensor:
        """computes symplectic integration step of input

        Args:
            dt (float): step size.
            x (torch.Tensor): (batchsize,dim)-input.

        Returns:
            x (torch.Tensor): (batchsize,dim)-output
            p_{n+1} = p_n-hf_p(p_n)
            q_{n+1} = q_n+hf(q_{n+1}).

        """
        n = x.shape[0]
        d = self.d
        dNew = self.dNew
        # splitting into p and q
        p = x[:, 0:dNew]
        q = x[:, dNew:d]
        p = torch.reshape(p, (n, dNew))
        q = torch.reshape(q, (n, dNew))

        if self.method == "symEuler":
            q_next = torch.add(q, +dt * self.p_dot(p))
            p_next = torch.add(p, -dt * self.q_dot(q_next))
        if self.method == "leapfrog":
            p_half = torch.add(p, -(dt / 2) * self.q_dot(q))
            q_next = torch.add(q, dt * self.p_dot(p_half))
            p_next = torch.add(p_half, -(dt / 2) * (self.q_dot(q_next)))
        if self.method == "secondOrder":
            q_next = torch.add(q, dt * p)
            p_next = torch.add(p, -dt * self.q_dot(p_next))
        x = torch.cat((p_next, q_next), 1)
        return x
