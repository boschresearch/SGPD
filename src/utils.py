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
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as inter
import torch
from scipy.integrate import odeint
from torch.autograd import Function, Variable, grad
from torch.utils.data import DataLoader, TensorDataset

DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
Parameter = torch.nn.Parameter

def plot_uncertainty(mean, var):
    plt.plot(mean)
    plt.plot(mean -var, 'r.')
    plt.plot(mean + var, 'r')
    plt.show()

def l2_error(xTrue, xApprox):
    """evaluate l2 error between xTrue and xApprox"""
    err = (xTrue - xApprox) ** 2
    err = torch.sqrt(err)
    return err


def euler_explicit(yStart, h, N, f, t):
    """explicit Euler integration"""
    yPred = np.zeros((N, np.size(yStart)))
    yPred[0, :] = yStart
    yCurrent = yStart
    for j in range(1, N):
        val = f(yCurrent)
        yCurrent = yCurrent + h * np.reshape(val, yCurrent.shape)
        yPred[j, :] = yCurrent
    return yPred


def pendulumEnergy(X):
    """energy for pendulum problem"""
    p = X[:, 0]
    q = X[:, 1]
    e = 6 * (1 - np.cos(q)) + 0.5 * p ** 2
    return e


def energyImplicit(X):
    """energy non-sepable Hamiltonian system"""
    e = (X[:, 0] ** 2 + 1) * (X[:, 1] ** 2 + 1) / 2
    return e


# class generateBatches(object):


def create_dataset(dataset: torch.Tensor, batchsize: int):
    """ create torch.DataLoader from subtrajectories

    Args:
        dataset ((N, dim)-torch.Tensor): subtrajectories.
        batchsize (int): batchsize.

    Returns:
        dl (torch.DataLoader): DataLoader.

    """

    numTrajectories = dataset.shape[0]
    indices = torch.arange(0, numTrajectories)
    indices = indices.reshape(numTrajectories, 1)
    ds = TensorDataset(dataset, indices)
    dl = DataLoader(ds, batchsize, shuffle=True)
    return dl


def preprocess_data(dataset: np.ndarray, length: int):
    """ preprocesses single trajectory in subtrajectories with moving horizon

    Args:
        dataset (np-ndarray): trajectory.
        length (int): length of subtrajectories.

    Returns:
        data ((N,dim)-torch.Tensor): dataset of all subtrajectories.
        initialValues (N-torch.Tensor): index of initial value.

    """

    # determine number of trajectoreis
    numTrajectories = len(dataset)
    data = []
    initialValues = []
    for i in range(numTrajectories):
        # evaluate trajectory
        trajectory = dataset[i]
        trajectorySize = trajectory.shape[0]
        trajectoryDim = trajectory.shape[1]
        # indices of initial values, goes in steps of length
        arg = np.arange(0, trajectorySize - length)
        for j in range(arg.shape[0]):
            # evaluate subtrajectory
            X = FloatTensor(trajectory[arg[j] : arg[j] + length + 1, :])
            X = torch.reshape(X, (1, length + 1, trajectoryDim))
            # evaluate initial values
            init = trajectory[arg[j] : arg[j] + 1, :]
            initialValues.append(torch.from_numpy(init))
            data.append(X)
    initialValues = torch.cat(initialValues, 0)
    data = torch.cat(data, 0)
    return data, initialValues


