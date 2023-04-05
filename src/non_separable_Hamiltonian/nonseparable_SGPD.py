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

"""script to reproduce results for non separable Hamiltonian system with midpoint-based SGPD"""

import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from torch.autograd import Function, Variable, grad
from nonseparable_utils import initialize_matrices, evaluate_model

sys.path.append("../")
import itertools

import scipy.interpolate as inter

from integrator.Implicit.IFT import AbstractProblem, Argmin
from integrator.nonsep_midpoint import ImplicitMidpoint
from sparse_GP.matherons_rule import MatheronDerivative
from utils import create_dataset, preprocess_data
from nonseparable_utils import initialize_matrices, plot_status, sample_model, sample_SGPD, det_from_list, plot_samples, plot_det, energy_computation

FloatTensor = torch.FloatTensor
DoubleTensor = torch.DoubleTensor
from Noise.noise_ND import GaussianNoise

Parameter = torch.nn.Parameter
import random

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

# load data
npzfile = np.load("nonsep_data.npz")
sorted(npzfile.files)
sol = npzfile["sol"]
test = npzfile["test"]
sol_ref = npzfile["solRef"]

# training settings
min_loss = np.infty
dt = 0.1
horizon = 10
d = 2
mode = "sample_predictions"
DS_trajectories = 5
ode = MatheronDerivative()
noiseLevel = torch.sqrt(torch.Tensor([[0.0005, 0.0005]]))
noise = GaussianNoise(2, noiselevel=noiseLevel, device="cpu")
params = [ode.parameters()]

# learning rate
opt = torch.optim.Adam(itertools.chain(*params), lr=1e-3)

# integrator settings
implicit_midpoint = ImplicitMidpoint(ode)
implicit_midpoint.method = "naive"
wrapped_implicit_midpoint = Argmin(implicit_midpoint)

datasize = sol.shape[0]
frac = np.floor(sol.shape[0] / horizon)

# initialize data collectors
ref_vals = FloatTensor(sol_ref)
steps = sol_ref.shape[0]
val = FloatTensor(sol)
val_steps = val.shape[0]
loss_matrix, epoch_loss, loss_matrix, val_loss, model_list = initialize_matrices()

data, initial_values = preprocess_data([sol], horizon)
data = data
dataset = create_dataset(data, 1)

# training
for c in range(0, 11):
    if c > 2:
       opt.param_groups[0]['lr'] = 0.0001
    if c > 5:
        opt.param_groups[0]['lr'] = 0.00001     
    if c % 1 == 0:
        print(c)
    if c == 10:
       print(c) 
    if val is not None:
        with torch.no_grad():
            X,_, vals = sample_SGPD(torch.Tensor(sol_ref[0, :]), 5, ode, wrapped_implicit_midpoint, mode)
            model_list.append(vals)
            loss = torch.sqrt(torch.sum((val - X[0:100,:]) ** 2) / 100)
            val_loss.append(loss.detach())
    epoch_curr = 0
    for batches in dataset:
        # loading training data
        opt.zero_grad()
        time_snippet = batches[0]
        indices = batches[1]
        # batches: [batchSize, horizon, d]"""
        x = time_snippet[0, 0, :].view(1, d) 
        X = [x]
        ode.mode = "sample_predictions" 
        sample_model(ode, mode)
        for h in range(horizon):
            x = wrapped_implicit_midpoint(x)
            X.append(x)
        X = torch.cat(X)
        Dist = noise(time_snippet[0, 1:, :], X[1:])
        # adapting contribution of KL according to length of training trajectory
        # compared to full trajectory
        loss = torch.sum(- Dist)+(ode.kl()/1e+6)
        loss.backward()
        opt.step()
        epoch_curr = epoch_curr + loss.detach()
        loss_matrix.append(loss.detach())
    epoch_loss.append(epoch_curr)
print("finished")

#model selection on lowest learning rate
index = torch.argmin(torch.Tensor(val_loss[7:]))
model = model_list[index+7]

evaluate_model(model, DS_trajectories, sol_ref, ref_vals, steps)


