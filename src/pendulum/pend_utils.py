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
import sys
sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Function, Variable, grad



#auxiliary functions for training 
def plot_samples(Y: torch.Tensor, sol_ref: torch.Tensor):
    with torch.no_grad():
        for i in range(Y.shape[0]):
            plt.plot(Y[i, :, :], label = "predictions")
        plt.xlabel("time", fontsize=15)
        plt.ylabel("states", fontsize=15)
        plt.legend(loc="upper right")
        plt.plot(sol_ref, label = "ground truth")
        plt.show()
     
def plot_status(X_mean: torch.Tensor, sol_ref: np.ndarray):
    plt.plot(X_mean[0, :, 0], label="prediction")
    plt.plot(sol_ref[:, 0], label="ground truth solution")
    plt.xlabel("time", fontsize=15)
    plt.ylabel("states", usetex=True, fontsize=15)
    plt.legend(loc="upper right")
    plt.show()
    err = (X_mean - sol_ref) ** 2
    l2 = np.sqrt(sum(sum(np.transpose(err))) / 400)
    print("L2-error of mean predictions: ", l2)
    
def plot_det(det_vals: torch.Tensor):
    """plot trajectory, ground truth and determinant values"""
    plt.plot(det_vals, label="determinant plot")
    plt.xlabel("time", fontsize=15)
    plt.ylabel("$det(\phi)$", usetex=True, fontsize=15)
    plt.legend(loc="upper right")
    plt.show()
    

def evaluate_model(model, ref_val, sol_ref, steps, DS_trajectories):
    """ plots results for pendulum """
    X_mean = torch.sum(model, 0) / DS_trajectories
    X_mean = X_mean.view(1, steps, 2)
    X_var = torch.sqrt(torch.sum((model - X_mean) ** 2, 0) / steps - 1)
    loss = torch.sqrt(torch.sum((ref_val - X_mean[0, :, :]) ** 2) / steps)
    plot_samples(model, sol_ref)
