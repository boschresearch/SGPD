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

"""script to reproduce results of pendulum with symplectic Euler-based SGPD"""

import itertools
import math
import sys
sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Function, Variable, grad

from integrator.symplectic_integrator import SymInt
from Noise.noise_ND import GaussianNoise
from sparse_GP.matherons_rule import MatheronGP
from training.training_pend import Training
from utils import create_dataset, preprocess_data
from pend_utils import plot_samples, plot_status, evaluate_model
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
Parameter = torch.nn.Parameter

np.random.seed(50)
torch.manual_seed(50)

# specify training details 
device = torch.device("cpu")
training_steps = 150
horizon = 10
batchsize = 1
lr = 1e-2
dt = 0.1
d = 2
DS_trajectories = 5

# load data
npzfile = np.load("pend_data.npz")
sorted(npzfile.files)
sol = npzfile["sol"]
test = npzfile["test"]
sol_ref = npzfile["solRef"]
n = sol.shape[0]
frac = np.floor(n / horizon)

#preprocess data 
dataset, init = preprocess_data([sol], horizon)
dataload = create_dataset(dataset, batchsize)

# GP specifications
psi_x = (torch.linspace(-5, 5, 9)).view(9, 1)
psi_y = (torch.linspace(-3, 3, 9)).view(9, 1)

p = MatheronGP(
    d=1,
    n=9,
    psi=psi_x,
    log_L=torch.log(torch.sqrt(torch.Tensor([[2.0]]))),   
    log_K = torch.log(1e-8 * torch.ones(9, dtype=torch.float)),
    log_sigma_f=torch.log(torch.Tensor([0.01])),
    device=device,
)

q = MatheronGP(
    d=1,
    n=9,
    psi=psi_y,
    log_L=torch.log(torch.sqrt(torch.Tensor([[2.0]]))),
    log_K = torch.log(1e-8 * torch.ones(9, dtype=torch.float)),
    log_sigma_f=torch.log(torch.Tensor([0.01])),
    device=device,
)

ref_val = FloatTensor(sol_ref).unsqueeze(0)
val = FloatTensor(sol).unsqueeze(0)

noise = GaussianNoise(d, torch.sqrt(torch.Tensor([[0.1]])), device=device)

# integrator details 
integrator = SymInt
method = "symEuler"
steps = sol_ref.shape[0]
mode = "sample_predictions"

# perform training
trainer = Training(dt, d, p, q, integrator, noise, training_steps, lr, horizon, batchsize, method, None)
p, q, lossMatrix, test_loss, epoch_loss, model, index = trainer.train(dataload, n, frac, ref_val, val, steps, mode)

#evaluate model
evaluate_model(model, ref_val, sol_ref, steps, DS_trajectories)
 
 
