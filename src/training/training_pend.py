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

"""class for perform training"""
import sys
sys.path.append('.')
import itertools

# from torchdiffeq import odeint as torch_odeint
import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Function, Variable, grad
from torch.utils.data import DataLoader

FloatTensor = torch.FloatTensor
Parameter = torch.nn.Parameter


class Training:

    """coordinating training for explicit integrators

    Args:
        dt (float): step size
        d (int): dimension
        p (torch.nn.Module): dynamics
        q (torch.nn.Module, optional): second part of dynamics
        integrator (torch.nn.Module): integrator
        noise (torch.nn.Module): provides noise and log prob
        trainingSteps (int): number of trainig epochs
        lr (float): learning rate
        horizon (int): length of subtrajectories
        batchsize (int): batchsize
        method (str): specifying integrator
        initializer (torch.nn.Module, optional): initialize subtrajectory

    Attributes:
        dt (float): step size
        d (int): dimension
        p (torch.nn.Module): dynamics
        q (torch.nn.Module, optional): second part of dynamics
        noise (torch.nn.Module): provides noise and log prob
        trainingSteps (int): number of trainig epochs
        lr (float): learning rate
        batchsize (int): batchsize        
        horizon (int): length of subtrajectories
        integrator (torch.nn.Module): integrator        
        method (str): specifying integrator
        initializer (torch.nn.Module, optional): initialize subtrajectory
    """

    def __init__(
        self,
        dt: float,
        d: int,
        p: torch.nn.Module,
        q: torch.nn.Module,
        integrator: torch.nn.Module,
        noise: torch.nn.Module,
        trainingSteps: int,
        lr: float,
        horizon: int,
        batchsize: int,
        method: str,
        initializer: torch.nn.Module,
    ):
        self.dt = dt
        self.d = d
        self.p = p
        self.q = q
        self.noise = noise
        self.trainingSteps = trainingSteps
        self.lr = lr
        self.batchsize = batchsize
        self.horizon = horizon
        self.integrator = integrator
        self.method = method
        self.initializer = initializer

    def prediction(self, x0: torch.Tensor, integrator: torch.nn.Module, steps: int, mode:str) -> torch.Tensor:
        """mean predictions with input x0"""
        xNew = x0.unsqueeze(1)
        X = [xNew]
        x = x0
        self.sample_model(mode)
        for h in range(steps -1):
            x = integrator(self.dt, x).detach()
            x_in = x.unsqueeze(1)
            X.append(x_in)
        X = torch.cat(X, 1)    
        return X 
        
    def initialize_matrices(self, n, test):
        """initialize matrices to collect training results""" 
        frac = np.floor(n / self.horizon)
        loss_matrix = []
        epoch_loss = []
        test_loss = []
        ref_loss = []
        model_list = []
        test_steps = test.shape[1]
        test_batch = test.shape[0]        
        return frac, loss_matrix, epoch_loss, test_loss, ref_loss, model_list, test_steps, test_batch
 
    def sample_model(self, mode):
        """ sample from model, i.e. update kernel at inducing inputs, draw inducing targets from distribution,
            sample weights from Matheron's rule""" 
        self.p.psi_kernel()
        if mode == "mean_preditions":
           self.p.mean_targets()
        else:   
           self.p.draw_targets()
        self.p.sample_basis_weights()
        self.p.mode = mode
        if self.q is not None:
            self.q.psi_kernel()
            if mode == "mean_predictions":
               self.q.mean_targets()
            else: 
               self.q.draw_targets()
            self.q.sample_basis_weights()
            self.q.mode = mode
 
    def define_integrator(self):
        """construct integrator from dimension, integration method and models""" 
        if self.q is not None:
            integrator = self.integrator(self.d, self.method, self.p, self.q)
        else:
            integrator = self.integrator(self.d, self.method, self.p)        
        return integrator 
    
    def define_params(self):
        """construct trainable parameters""" 
        if self.q is not None:
            params = [self.p.parameters(), self.q.parameters()]
        else:
            params = [self.p.parameters()]
    
        opt = torch.optim.Adam(itertools.chain(*params), lr=self.lr)
        return opt
    
    def train(
        self, dataset: DataLoader, n: int, frac: float, ref_val: torch.Tensor, test: torch.Tensor, steps: int, mode:str 
    ) -> Tuple[torch.nn.Module, torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        """train dynamics

        Args:
            dataset (DataLoader): subtrajectories in dataloader.
            n (int): number of training steps.
            frac (float): weighting factor kl divergence.
            test ((batchsize, d, N)-torch.Tensor): test data.
            steps (int): number of rollout steps for test data
            mode (str): rollout mode for test data ("mean_predictions" or "sample_predictions")
        Returns:
            p (torch.nn.Module): trained dynamics.
            q (torch.nn.Module): trained dynamics.
            loss_matrix (torch.Tensor): loss over epochs.
            val_loss (torch.Tensor): test loss for test.
            epoch_loss (torch.Tensor): summed loss over epochs. 
            model ((5, steps, d)-torch.Tensor): predictions obtained from model selection 
            index (int): index for model selection 
        """
        frac, loss_matrix, epoch_loss, test_loss, ref_loss, model_list, test_steps, test_batch = self.initialize_matrices(n, test)
        # defining the training parameters
        integrator = self.define_integrator()
        opt = self.define_params()
        # training
        for c in range(0, self.trainingSteps):
            epoch_curr = 0    
            if c % 1 == 0:
                print(c)
                """ test data
                """
                if test is not None:
                      #predifine matrix
                      Y = torch.zeros((5, steps, self.d))
                      x0 = test[:, 0, :]
                      self.p.reset_local_model()
                      if self.q is not None:
                         self.q.reset_local_model()
                      for i in range(5):
                          self.sample_model(mode)                  
                          X = self.prediction(x0, integrator, steps, mode)                           
                          self.p.save_local_model()
                          if self.q is not None:
                             self.q.save_local_model()                                 
                          Y[i, :, :] = X
                      self.p.save_global_model()
                      if self.q is not None:
                         self.q.save_global_model() 
                      model_list.append(Y)          
                      X_mean= torch.sum(Y, 0) / 5
                      X_mean = X_mean.unsqueeze(0)
                      loss = torch.sqrt(
                          torch.sum((test[:, 1:, :] - X_mean[:, 1 : test.shape[1], :]) ** 2) / (test.shape[1] - 1)
                        )
                      test_loss.append(loss.detach())
                         
            for batches in dataset:
                """ loading training data"""
                opt.zero_grad()
                time_snippet = batches[0]
                indices = batches[1]
                # batches: [batchSize, horizon, d]"""
                x = time_snippet[:, 0, :]
                current_batchsize = indices.shape[0]
                x = torch.reshape(x, (current_batchsize, self.d))
                x_new = torch.reshape(x, (current_batchsize, 1, self.d))
                X = [x_new]
                self.sample_model(mode)                  
                for h in range(self.horizon):
                    x = integrator(self.dt, x)
                    xNew = x.view((current_batchsize, 1, self.d))
                    X.append(xNew)
                X = torch.cat(X, 1)
                kl = self.p.kl()
                if self.q is not None:
                    kl = kl + self.q.kl()
                # \sum_n log(p(\hat{x}_n|x_n))
                Dist = self.noise(X[:, 1:, :], time_snippet[:, 1:, :])
                # adapting contribution of KL according to length of training trajectory
                loss = 2 * (-frac) * Dist +(kl/1e+6)
                loss.backward()
                opt.step()
                epoch_curr = epoch_curr + loss.detach()
                loss_matrix.append(loss.detach())
            epoch_loss.append(epoch_curr)
        print("finished")
        index = torch.argmin(torch.Tensor(test_loss))
        model = model_list[index]
        return self.p, self.q, loss_matrix, test_loss, epoch_loss, model, index
    
    def predict_from_list(self, x0, index, steps, N_samples):
        """
        draws prediction from model selection 
        
        Parameters
        ----------
        x0 : (1,d)-torch.Tensor
            initial value of trajectory.
        index : int
            index for model selection.
        steps : int
            rollout steps.
        N_samples : int
            number of samples.

        Returns
        -------
        Y : (N_samples, steps, d)-torch.Tensor
            rollout with specified number of steps

        """
        integrator = self.define_integrator()
        Y = torch.zeros((N_samples, steps, self.d))
        xNew = x0.view((1, 1, self.d))
        for i in range(N_samples):
            self.p.load_model(index, i)
            if self.q is not None:
                self.q.load_model(index, i)
            x = x0.view((1, self.d))
            X = [xNew]
            for h in range(steps -1):
                x = integrator(self.dt, x).detach()
                x_in = x.view((1, 1, self.d))
                X.append(x_in)
            X = torch.cat(X, 1)    
            Y[i, :, :] = X
        return Y      

    def det_from_list(self, x0, index, steps, N_samples):
        """
        draws prediction from model selection 
        
        Parameters
        ----------
        x0 : (1,d)-torch.Tensor
            initial value of trajectory.
        index : int
            index for model selection.
        steps : int
            rollout steps.
        N_samples : int
            number of samples.

        Returns
        -------
        Y : (steps, d)-torch.Tensor
            determinants of explicit Euler integrator along trajectory 

        """
        integrator = self.define_integrator()
        Y = torch.zeros((N_samples, steps-1))
        xNew = x0.view((1, 1, self.d))
        for i in range(N_samples):
            self.p.load_model(index, i)
            x = x0.view((1, self.d))
            det_vals = []
            for h in range(steps -1):
                det_val = self.p.det(x, self.dt)
                det_vals.append(det_val.unsqueeze(0))                
                x = integrator(self.dt, x).detach()
            det_vals = torch.cat(det_vals)    
            Y[i, :] = det_vals
        return Y    
    
 