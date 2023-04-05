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

"""forward and backward pass for implicit layer via implicit function theorem"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class AbstractProblem(torch.nn.Module, ABC):
    def __init__(self,):
        super().__init__()
        self.method = "naive"
        self._jitter = 1e-3
        self._rtol = 1e-2

    def forward(self, u: torch.Tensor):
        """Evaluate some random input
        """
        pass

    @abstractmethod
    def solve(self):
        """Solve for optimal solution
        """
        pass


class Argmin(torch.nn.Module):
    """Wrapping up the Actual armgin code in some PyTorch.nn Functionality
    """

    def __init__(self, problem: AbstractProblem):
        super().__init__()
        self.problem = problem

    def forward(self, x: Optional[torch.Tensor] = None):
        return ArgminFunction.apply(self.problem, x, *self.problem.parameters())


class ArgminFunction(torch.autograd.Function):
    """Providing backward via IFT

    f = problem instance
    x = optional inputs
    *args = f.parameters()
    """

    @staticmethod
    def forward(ctx, f: AbstractProblem, x: Tensor, *args):
        # solve functionality is supplied by problem.
        # see https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483
        if x is not None:
            ctx.x = x.detach().clone().requires_grad_(x.requires_grad)
        else:
            ctx.x = None
        ctx.y_star = u.view(-1).detach().clone().requires_grad_(True)
        ctx.f = f

        return u  # this is y_star

    @staticmethod
    def _get_aug_grad_naive(d_y, y_star, grad_output):
        dim = grad_output.shape[0]

        d_yy = []
        for c in range(dim):
            # this loop is the bottleneck
            # maybe distributed pytorch can help here
            torch.set_grad_enabled(True)
            d_yy.append(torch.autograd.grad(d_y[c], y_star, retain_graph=True)[0])

        d_yy = torch.cat(d_yy).view(dim, dim)
        d_yy_inv = torch.inverse(d_yy)
        aug_grad = -grad_output @ d_yy_inv
        return aug_grad

    @staticmethod
    def backward(ctx, grad_output):
        # unpack stored information from forward call
        y_star = ctx.y_star
        x = ctx.x
        f = ctx.f
        # flatten grad_output
        grad_output = grad_output.view(-1)

        # problem input dependent?
        x_grad = False
        if x is not None:
            if x.requires_grad:
                x_grad = True
        torch.set_grad_enabled(True)  # direct access C code
        f_eval = f(y_star, x)
        d_y, *_ = torch.autograd.grad(f_eval, y_star, create_graph=True, retain_graph=True)
        # calc grad_output@Hesian_inverse
        if f.method == "naive":
            aug_grad = ArgminFunction._get_aug_grad_naive(d_y, y_star, grad_output)
        grad_theta = torch.autograd.grad(d_y, f.parameters(), aug_grad, retain_graph=x_grad)

        if x_grad:
            grad_x = torch.autograd.grad(d_y, x, aug_grad, retain_graph=False)[0]
        else:
            grad_x = None
        return_tuple = (None, grad_x, *grad_theta)
        return return_tuple