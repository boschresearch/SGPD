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

import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from torch.autograd import Function, Variable, grad

sys.path.append("../")
import itertools

import scipy.interpolate as interpolate

FloatTensor = torch.FloatTensor
DoubleTensor = torch.DoubleTensor

Parameter = torch.nn.Parameter
import random
from typing import Tuple

def sample_model(ode:torch.nn.Module, mode:str):
    """sample parameters from ode""" 
    ode.psi_kernel()
    if mode == "mean_targets":
       ode.mean_targets()
    else:   
       ode.draw_targets() 
    ode.sample_basis_weights()    

def initialize_matrices():
    """initialize losses""" 
    loss_matrix = []
    epoch_loss = []
    loss_matrix = []
    test_loss = []
    model_list = []
    return loss_matrix, epoch_loss, loss_matrix, test_loss, model_list

def plot_samples(Y:torch.nn.Module, sol_ref:np.ndarray):
    """pltos samples and ground truth""" 
    with torch.no_grad():
        for i in range(Y.shape[0]):
            plt.plot(Y[i, :, :], label = "predictions")
        plt.xlabel("time", fontsize=15)
        plt.ylabel("states", fontsize=15)
        plt.plot(sol_ref, label = "ground truth")
        plt.legend(loc="upper right")
        plt.show()
        
def det_from_list( x0:torch.Tensor, ode:torch.nn.Module, index:int, steps:int, N_samples:int, dt:float):
    """
    computes determinants from model seletion  
    
    Parameters
    ----------
    x0 : (1,d)-torch.Tensor
        initial value of trajectory.
    ode: torch.nn.Module
         dynamics     
    index : int
        index for model selection.
    steps : int
        rollout steps.
    N_samples : int
        number of samples.
    dt: float
        step size
    Returns
    -------
    Y : (steps, d)-torch.Tensor
        determinants of explicit Euler integrator along trajectory 

    """
    Y = torch.zeros((N_samples, steps-1))
    xNew = x0.view((1, 1, 2))
    for i in range(N_samples):
        ode.load_model(index, i)
        x = x0.view((1, 2))
        det_vals = []
        for h in range(steps -1):
            det_val = ode.det(x, dt)
            det_vals.append(det_val.unsqueeze(0))                
            x = x + dt * ode(x)
        det_vals = torch.cat(det_vals)    
        Y[i, :] = det_vals
    return Y    
  
def sample_SGPD(x0: np.ndarray, N: int, ode:torch.nn.Module, integrator:torch.nn.Module, mode:str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sampels N predictions with from ode with integrator 

    Parameters
    ----------
    x0 : torch.Tensor
        initial value.
    N : int
        samples.
    ode : torch.nn.Module
        ode dynamics.
    integrator : torch.nn.Module
        integrator.
    mode : str out of "sample_predictions", "mean_predictions"
         type of rollouts.

    Returns
    -------
    mean : (horizon, dim)-torch.Tensor
        mean of smaples.
    var : (horizon, dim)-torch.Tensor
        std deviation of samples.
    vals : (N,horizon,dim)-torch.Tensor
        samples.

    """
    
    vals = torch.Tensor(N, 400, 2)
    energy = torch.Tensor(N, 400)
    with torch.no_grad():
         for i in range(N):
             x_new = x0.view((1, 1, 2))
             x = x_new.view((1,2))
             X = [x_new]
             ode.set_mode(mode)
             sample_model(ode, mode)
             for h in range(400 - 1):
                 x = integrator(x)
                 x_new = x.view((1, 1, 2))
                 X.append(x_new)
             X = torch.cat(X, 1)
             vals[i, :, :] = X
         mean = torch.sum(vals, 0)/ N
         var = torch.sqrt(torch.sum((vals- mean)**2, 0)/ (N-1))
         return mean, var, vals
       

def sample_Euler(x_0: torch.Tensor, N:int, mode:str, ode:torch.nn.Module, dt:float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples predictions with Euler method and saves the models 

    Parameters
    ----------
    x_0 : torch.Tensor
        initial value.
    N : int
        number of samples.
    mode : str our of "sample_predictions", "mean_prediction"
        indicates, whether predictions are sampled or mean-predictions are conducted 
    ode : torch.nn.Module
        ode dynamics.
    dt : float
        step size.

    Returns
    -------
    X_mean : (horizon, dimension)- torch.Tensor
        mean of samples.
    X_var : (horizon, dimension)- torch.Tensor
        std deviation of samples.
    Y : (N, horizon, dimension)-torch:Tensor
        all samples.

    """
    Y = torch.zeros(N, 400, 2)
    with torch.no_grad():
        ode.reset_local_model()
        for i in range(N):
            ode.set_mode(mode)
            sample_model(od, mode)
            x_new = x_0.view((1, 1, 2))
            X = [x_new]
            x = x_new.view((1, 2))
            for h in range(400 - 1):
                x = x + dt * ode(x)
                x_new = x.view((1, 1, 2))
                X.append(x_new)
            X = torch.cat(X, 1)
            Y[i, :, :] = X
            ode.save_local_model()
        ode.save_global_model()    
        X_mean = torch.sum(Y, 0)/N
        X_var = torch.sqrt(torch.sum((Y - X_mean)**2, 0)/(N-1))
    return X_mean, X_var, Y

def plot_det(det_vals: torch.Tensor):
    """plot trajectory, ground truth and determinant values"""
    plt.plot(det_vals.t(), label="determinant plot")
    plt.xlabel("time", fontsize=15)
    plt.ylabel("$det(\phi)$", usetex=True, fontsize=15)
    plt.legend(loc="upper right")
    plt.show()
        

def plot_status(det_vals: torch.Tensor):
    """plot prediction, ground truth, and determinant"""
    plt.plot(det_vals, label="determinant plot")
    plt.xlabel("time", fontsize=15)
    plt.ylabel("$det(\phi)$", usetex=True, fontsize=15)
    plt.legend(loc="upper right")
    plt.show()

        
def energy_computation(sol_ref:np.ndarray, model:torch.Tensor):
    """computes energy mean and energy std deviation compared to ground truth""" 
    eRef = 0.5*(1+sol_ref[:,0]**2)*(1+sol_ref[:,1]**2)
    e_SGPD = 0.5*(1+model[:, :,0]**2)*(1+model[:, :, 1]**2)
    e_mean = torch.mean(e_SGPD, 0)
    total_e_mean = torch.mean(e_mean)
    e_var = torch.sqrt(torch.sum((e_mean-eRef[0])**2)/400)     
    print("energy mean: ", total_e_mean)
    print("energy_var: ", e_var)


def evaluate_model(model, DS_trajectories, sol_ref, ref_vals, steps):
    """ evaluates and plots model"""
    mean = torch.sum(model, 0) / DS_trajectories
    var = torch.sqrt(torch.sum((model - mean) ** 2, 0) / (DS_trajectories - 1))
    loss = torch.sqrt(torch.sum((ref_vals - mean) ** 2) / steps)
    plot_samples(model, sol_ref)
