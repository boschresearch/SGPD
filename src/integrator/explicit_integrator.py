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

"""integrator classes"""

import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as inter
import torch
from scipy.integrate import odeint, solve_ivp
from torch.autograd import Function, Variable, grad

FloatTensor = torch.FloatTensor
Parameter = torch.nn.Parameter

from typing import Optional


class ExplicitSolver(torch.nn.Module):
    """some explicit integrators for torch.nn.module

    Args:
        d (int, optional): dimension of problem. Defaults to 2.
        method (str, optional): specification of integrator. Defaults to "explicitEuler".
        ode (torch.nn.module, optional): dynamics velocity

    Attributes:
        d (int, optional): dimension of problem. Defaults to 2.
        method (str, optional): specification of integrator. Defaults to "explicitEuler".
        ode (torch.nn.module, optional): dynamics velocity
    """

    def __init__(self, d: int = 2, method: str = "explicitEuler", ode: Optional[torch.nn.Module] = None):

        super().__init__()
        self.d = d
        self.ode = ode
        self.method = method

    def forward(self, h: float, x: torch.Tensor) -> torch.Tensor:
        """integrates input with specified step size

        Args:
            h (float): stepsie.
            x (torch.Tensor): (batchsize,d)-input.

        Returns:
            x (torch.tensor): (batchsizue,d)-output.

        """
        d = self.d
        self.update = False
        if self.method == "explicitEuler":
            f = self.ode(x)
            x = torch.add(x, h * f)
        if self.method == "Heun":
            """
            k1 = f(x)
            k2 = f(x+hk1)
            x_{n+1}=x_n+(h/2)(k1+k2)
            """
            k1 = self.ode(x)
            x1 = x + torch.mul(h, k1)
            k2 = self.ode(x1)
            x = x + (h / 2) * (torch.add(k1, k2))
        if self.method == "RK3":
            """
            k1 = f(x)
            k2 = f(x+(h/2)k1)
            k3 = f(x-h*(k1+2*h*k2))
            x_{n+1}=x_n+(h/6)(k1+4*k2+k3)
            """
            k1 = self.ode(x)
            x1 = torch.add(x, (h / 2) * k1)
            k2 = self.ode(k1)
            x2 = torch.add(x, -h * torch.add(k1, 2 * h * k2))
            k3 = self.ode(x2)
            x = x + (h / 6) * (torch.add(k1, torch.add(4 * k2, k3)))
        if self.method == "RK4":
            """
            k1 = f(x)
            k2 = f(x+(h/2)k1)
            k3 = f(x+(h/2)k2)
            k4 = f(x+(h/2)k3)
            x_{n+1}=x_n+(h/6)(k1+k2+k3+k4)

            """
            k1 = self.ode(x)
            x1 = torch.add(x, (h / 2) * k1)
            k2 = self.ode(x1)
            x2 = torch.add(x, (h / 2) * k2)
            k3 = self.ode(x2)
            x3 = torch.add(x, h * k3)
            k4 = self.ode(x3)
            x = torch.add(x, (h / 6) * torch.add(2 * k1, torch.add(k2, torch.add(k3, 2 * k4))))
        return x
